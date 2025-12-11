from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.api.schemas import AnalyzeRequest, AnalyzeResponse
from app.adapters.github.adapter import GitHubAdapter
from app.core.agents.builder import build_sirius_graph
from app.core.database.session import create_db_and_tables
from app.core.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On Startup
    create_db_and_tables()
    yield
    # On Shutdown

app = FastAPI(
    lifespan=lifespan,
    title="Sirius Compass API",
    description="Engineering Intelligence Platform Backend",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "sirius-compass"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repository(request: AnalyzeRequest):
    """
    Triggers the LangGraph workflow to analyze a repository.
    """
    logger.info(f"🚀 API Request: Analyze {request.repo_url}")
    
    try:
        # 1. Parse Repo Name
        # Basic parsing, assumes https://github.com/owner/repo format
        parts = request.repo_url.rstrip("/").split("/")
        if len(parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid GitHub URL")
        repo_name = f"{parts[-2]}/{parts[-1]}"
        
        # 2. Ingest Data (Adapter)
        # Note: GitHubAdapter is synchronous, but optimized with internal threads.
        # In a high-load scenario, we would offload this to a background task (Celery/Arq).
        adapter = GitHubAdapter()
        activities = adapter.fetch_recent_activity(repo_name, days=request.lookback_days)
        
        if not activities:
            return {
                "status": "warning", 
                "report": {}, 
                "metadata": {"message": "No recent activity found"}
            }

        # 3. Execute Agent Workflow (Core)
        initial_state = {
            "repo_name": repo_name,
            "developer_name": request.developer_name,
            "activities": activities,
            "analysis_logs": [],
            "final_report": None
        }
        
        workflow = build_sirius_graph()
        # invoke() is synchronous. LangGraph supports astream() for streaming (Next Step).
        result = workflow.invoke(initial_state)
        
        final_report = result.get("final_report")
        
        return {
            "status": "success",
            "report": final_report.dict() if final_report else {},
            "metadata": {"activities_processed": len(activities)}
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# To run: poetry run uvicorn app.api.server:app --reload