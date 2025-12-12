from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from sqlmodel import Session
from contextlib import asynccontextmanager

# Import Internal Modules
from app.api.schemas import AnalyzeRequest, AnalyzeResponse
from app.core.database.session import create_db_and_tables, get_session
from app.services.sync import SyncService
from app.core.agents.builder import build_sirius_graph
from app.core.models.domain import UnifiedActivity, ActivityType
from app.core.logger import get_logger

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

# 2. Mount Static Files (For Charts/Multimodality)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "sirius-compass"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repository(
    request: AnalyzeRequest, 
    session: Session = Depends(get_session) # Inject DB Session
):
    """
    Triggers the analysis workflow.
    Architecture: API -> SyncService -> DB (Store) -> DB (Read) -> Agent -> Response
    """
    logger.info(f"🚀 API Request: Analyze {request.repo_url}")
    
    try:
        # 1. Parse Repo Name
        parts = request.repo_url.rstrip("/").split("/")
        if len(parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid GitHub URL")
        repo_name = f"{parts[-2]}/{parts[-1]}"
        
        # 2. Smart Sync (Data Lake Ingestion)
        # This handles GitHub + Linear (if configured)
        sync_service = SyncService(session)
        logger.info(f"🔄 Syncing repository data for {repo_name}...")
        repo_db = sync_service.ensure_repository_sync(request.repo_url, request.lookback_days)
        
        # 3. Retrieve Data from DB (Local Access)
        db_activities = repo_db.activities
        
        if not db_activities:
            return {
                "status": "warning", 
                "report": {}, 
                "metadata": {"message": "No activity found in database after sync"}
            }

        # 4. Map SQL Models -> Domain Models (for the Agent)
        domain_activities = []
        for db_act in db_activities:
            # We map only what the Agent needs for code analysis
            # (Linear tickets might be skipped here if the Agent focuses only on code, 
            # or included if we want to compare Ticket vs Code)
            try:
                domain_act = UnifiedActivity(
                    source_id=db_act.source_id,
                    source_platform=db_act.source_platform,
                    type=ActivityType(db_act.type) if db_act.type in ActivityType.__members__ else ActivityType.COMMIT, 
                    author=db_act.author,
                    content=db_act.content,
                    timestamp=db_act.timestamp,
                    url=f"{request.repo_url}/blob/main/{db_act.source_id}", 
                    files_changed=[] 
                )
                domain_activities.append(domain_act)
            except ValueError:
                # Fallback for types not in the Enum (like TICKET)
                pass

        # Sort by timestamp desc
        domain_activities.sort(key=lambda x: x.timestamp, reverse=True)

        # 5. Execute Agent Workflow (Core)
        initial_state = {
            "repo_name": repo_name,
            "developer_name": request.developer_name,
            "activities": domain_activities,
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
                "source": "database_cache"
            }
        }

    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")