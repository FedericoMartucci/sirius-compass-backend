from fastapi import FastAPI, HTTPException, Depends
from dotenv import load_dotenv
from sqlmodel import Session
from app.api.schemas import AnalyzeRequest, AnalyzeResponse
from app.core.database.session import create_db_and_tables, get_session
from app.services.sync import SyncService
from app.core.agents.builder import build_sirius_graph
from app.core.models.domain import UnifiedActivity, ActivityType
from app.core.logger import get_logger
from contextlib import asynccontextmanager

# 1. Load environment variables first!
load_dotenv() 

logger = get_logger(__name__)

# Lifespan event to initialize the DB on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🏁 System Startup: Initializing Database...")
    create_db_and_tables()
    yield
    logger.info("🛑 System Shutdown")

app = FastAPI(
    title="Sirius Compass API",
    description="Engineering Intelligence Platform Backend",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "sirius-compass"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repository(
    request: AnalyzeRequest, 
    session: Session = Depends(get_session) # Injection of DB Session
):
    """
    Triggers the LangGraph workflow to analyze a repository.
    Uses 'SyncService' to ensure data is fresh in the DB before analyzing.
    """
    logger.info(f"🚀 API Request: Analyze {request.repo_url}")
    
    try:
        # 1. Parse Repo Name
        parts = request.repo_url.rstrip("/").split("/")
        if len(parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid GitHub URL")
        repo_name = f"{parts[-2]}/{parts[-1]}"
        
        # 2. Smart Sync (The "Data Lake" Strategy)
        # Instead of calling the Adapter directly, we call the Service.
        # This handles: Incremental fetching, DB saving, and duplicate prevention.
        sync_service = SyncService(session)
        logger.info(f"🔄 syncing repository data for {repo_name}...")
        repo_db = sync_service.ensure_repository_sync(request.repo_url, request.lookback_days)
        
        # 3. Retrieve Data from DB (Local Access = Milliseconds)
        # We fetch the activities stored in our SQL DB.
        db_activities = repo_db.activities
        
        if not db_activities:
            return {
                "status": "warning", 
                "report": {}, 
                "metadata": {"message": "No activity found in database after sync"}
            }

        # 4. Map SQL Models -> Domain Models
        # The Agent expects 'UnifiedActivity' (Pydantic), but DB gives 'Activity' (SQLModel).
        # We must map them.
        domain_activities = []
        for db_act in db_activities:
            # Simple conversion
            domain_act = UnifiedActivity(
                source_id=db_act.source_id,
                source_platform="github", # Hardcoded for now, dynamic in future
                type=ActivityType(db_act.type), # Convert string back to Enum
                author=db_act.author,
                content=db_act.content,
                timestamp=db_act.timestamp,
                url=f"{request.repo_url}/commit/{db_act.source_id}", # Reconstruct URL
                files_changed=[] # We stored count, not list in DB (Optimization choice)
            )
            domain_activities.append(domain_act)

        # Sort by timestamp desc to give Agent the latest context
        domain_activities.sort(key=lambda x: x.timestamp, reverse=True)

        # 5. Execute Agent Workflow (Core)
        initial_state = {
            "repo_name": repo_name,
            "developer_name": request.developer_name,
            "activities": domain_activities, # Passing mapped objects
            "analysis_logs": [],
            "final_report": None
        }
        
        workflow = build_sirius_graph()
        result = workflow.invoke(initial_state)
        
        final_report = result.get("final_report")
        
        report_data = final_report.dict() if final_report else {}

        return {
            "status": "success",
            "report": report_data,
            "metadata": {
                "activities_processed": len(domain_activities),
                "source": "database_cache" # Indicator that data came from SQL
            }
        }

    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")