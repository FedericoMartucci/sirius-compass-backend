# app/api/server.py
import os
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Import Schemas from your existing file
from app.api.schemas import AnalyzeRequest, AnalyzeResponse, ChatRequest

from app.core.database.session import create_db_and_tables
from app.core.agents.analyst_graph import build_analyst_graph
from app.core.agents.chat_graph import build_chat_graph
from app.core.logger import get_logger

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
        
        # 2. Build Graph
        workflow = build_analyst_graph()
        
        # 3. Initial State
        initial_state = {
            "repo_name": repo_name,
            "developer_name": request.developer_name,
            "lookback_days": request.lookback_days,
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
async def chat_agent(request: ChatRequest):
    """
    Talks to the Conversational Graph.
    """
    try:
        workflow = build_chat_graph()
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Inject Context into User Message
        input_message = {
            "messages": [("user", f"[Context: Repo {request.repo_name}] {request.message}")]
        }
        
        result = await workflow.ainvoke(input_message, config=config)
        last_message = result["messages"][-1]
        
        return {
            "response": last_message.content,
            "thread_id": request.thread_id
        }
    except Exception as e:
        logger.error(f"Chat Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))