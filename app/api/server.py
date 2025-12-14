# app/api/server.py
import asyncio
import os

from fastapi import Request
from fastapi.responses import StreamingResponse
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from sqlmodel import Session

# Import Schemas from your existing file
from app.api.schemas import AnalyzeRequest, AnalyzeResponse, ChatRequest

from app.core.database.session import create_db_and_tables, engine
from app.core.agents.analyst_graph import build_analyst_graph
from app.core.agents.chat_graph import build_chat_graph
from app.core.logger import get_logger
from app.core.streaming import TokenStreamHandler, sse_data
from app.services.chat_storage import (
    append_message,
    coerce_content_to_text,
    get_or_create_thread,
    load_thread_messages,
)

load_dotenv()
logger = get_logger(__name__)

# --- Helper: Robust URL Parsing (Kept here as internal util) ---
def _parse_repo_name(url_str: str) -> str:
    """
    Extracts 'owner/repo' safely from URL.
    """
    clean_url = url_str.strip().rstrip("/")
    if not clean_url.startswith("http") and "/" in clean_url:
        parts = clean_url.split("/")
        if len(parts) == 2: return f"{parts[0]}/{parts[1]}"
            
    try:
        parsed = urlparse(clean_url)
        path = parsed.path.strip("/")
        if path.endswith(".git"): path = path[:-4]
        parts = path.split("/")
        if len(parts) < 2: raise ValueError("URL path too short")
        return f"{parts[-2]}/{parts[-1]}"
    except Exception:
        raise ValueError(f"Could not parse repository name from: {url_str}")

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🏁 System Startup: Initializing Database...")
    create_db_and_tables()
    yield
    logger.info("🛑 System Shutdown")

app = FastAPI(title="Sirius Compass API", lifespan=lifespan)

# --- Static Files ---
if not os.path.exists("static"): os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Endpoint 1: ANALYST (Batch) ---
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repo(request: AnalyzeRequest):
    """
    Triggers the heavy analysis graph (GitHub + Linear -> Report -> DB).
    """
    logger.info(f"🚀 Starting Analysis for {request.repo_url} (Lookback: {request.lookback_days} days)")
    
    try:
        # 1. Parse Repo Name
        repo_name = _parse_repo_name(request.repo_url)
        project_name = request.project_name or repo_name
        
        # 2. Build Graph
        workflow = build_analyst_graph()
        
        # 3. Initial State
        initial_state = {
            "repo_name": repo_name,
            "project_name": project_name,
            "developer_name": request.developer_name,
            "lookback_days": request.lookback_days,
            "linear_team_key": request.linear_team_key,
            "activities": [],
            "analysis_logs": [],
            "final_report": None
        }
        
        # 4. Invoke
        result = await workflow.ainvoke(initial_state)
        report = result.get("final_report")
        
        # Construct Response using Schema
        return AnalyzeResponse(
            status="success",
            message="Analysis complete and saved to DB.",
            report_summary=report.feedback_summary if report else "No report generated.",
            report=report.dict() if report else {}, # Fallback
            metadata={"repo": repo_name}
        )
        
    except ValueError as ve:
        logger.error(f"Input Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Analysis Failed: {e}")
        # Return generic error but log full trace
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# --- Endpoint 2: CHAT (Interactive) ---
@app.post("/chat")
async def chat_agent(payload: ChatRequest, http_request: Request):
    """
    Talks to the Conversational Graph.
    """
    try:
        project_name = payload.project_name or payload.repo_name
        with Session(engine) as session:
            thread = get_or_create_thread(
                session,
                external_thread_id=payload.thread_id,
                owner_id=payload.user_id,
            )
            thread_db_id = thread.id
            history = load_thread_messages(session, thread_db_id, limit=50)
            append_message(
                session,
                chat_thread_id=thread_db_id,
                role="user",
                content=payload.message,
                metadata={"repo_name": payload.repo_name, "project_name": project_name},
            )

        # Add an ephemeral system context message for the current request.
        messages = [
            ("system", f"Context: Project {project_name}. Repo {payload.repo_name}"),
            *history,
            ("user", payload.message),
        ]
        workflow = build_chat_graph()

        wants_stream = "text/event-stream" in (http_request.headers.get("accept") or "").lower()
        if wants_stream:
            async def event_generator():
                handler = TokenStreamHandler()
                task = asyncio.create_task(
                    workflow.ainvoke(
                        {"messages": messages},
                        config={"callbacks": [handler]},
                    )
                )

                try:
                    while True:
                        if task.done() and handler.queue.empty():
                            break
                        try:
                            token = await asyncio.wait_for(handler.queue.get(), timeout=0.1)
                            yield sse_data({"type": "token", "value": token})
                        except asyncio.TimeoutError:
                            continue

                    result = await task
                    last_message = result["messages"][-1]
                    assistant_text = coerce_content_to_text(last_message.content)

                    with Session(engine) as session:
                        append_message(
                            session,
                            chat_thread_id=thread_db_id,
                            role="assistant",
                            content=assistant_text,
                            metadata={"repo_name": payload.repo_name, "project_name": project_name},
                        )

                    yield sse_data({"type": "done", "thread_id": payload.thread_id})
                except Exception as e:
                    yield sse_data({"type": "error", "message": str(e)})
                    raise

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        result = await workflow.ainvoke({"messages": messages})
        last_message = result["messages"][-1]
        assistant_text = coerce_content_to_text(last_message.content)

        with Session(engine) as session:
            append_message(
                session,
                chat_thread_id=thread_db_id,
                role="assistant",
                content=assistant_text,
                metadata={"repo_name": payload.repo_name, "project_name": project_name},
            )

        return {"response": assistant_text, "thread_id": payload.thread_id}
    except Exception as e:
        logger.error(f"Chat Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
