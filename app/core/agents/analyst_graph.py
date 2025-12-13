# app/core/agents/analyst_graph.py
import asyncio
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from app.core.agents.state import GraphState
from app.core.agents.nodes import analyze_activities_node, generate_report_node
from app.adapters.github.adapter import GitHubAdapter
from app.adapters.linear.adapter import LinearAdapter
from app.core.models.domain import UnifiedActivity, ActivityType
from app.core.database.session import get_session
from app.core.database.models import AnalysisReport, Repository, Activity
from sqlmodel import select
from app.core.logger import get_logger

logger = get_logger(__name__)

# --- NODE 1: PARALLEL INGESTION ---
async def ingest_data_node(state: GraphState):
    """
    Fetches data from GitHub and Linear simultaneously using asyncio.
    """
    repo_name = state["repo_name"]
    # FIX: Leer los días del estado, o usar 90 por defecto.
    days = state.get("lookback_days", 90) 
    
    logger.info(f"🚀 Starting Parallel Ingestion for: {repo_name} (Lookback: {days} days)")

    gh_adapter = GitHubAdapter()
    lin_adapter = LinearAdapter()

    # Create async tasks (wrap synchronous calls in threads)
    # FIX: Pasamos la variable 'days' aquí
    task_gh = asyncio.to_thread(gh_adapter.fetch_recent_activity, repo_name, days)
    task_lin = asyncio.to_thread(lin_adapter.fetch_recent_issues, 50)

    # Execute in parallel
    results = await asyncio.gather(task_gh, task_lin)
    gh_data, lin_data = results[0], results[1]

    # Map to UnifiedActivity Domain Model
    activities = []
    
    # Map GitHub
    for item in gh_data:
        activities.append(UnifiedActivity(
            source_id=item.source_id, source_platform="github",
            type=item.type, author=item.author,
            content=item.content, timestamp=item.timestamp,
            url=f"https://github.com/{repo_name}/commit/{item.source_id}",
            files_changed=item.files_changed
        ))

    # Map Linear
    for item in lin_data:
        try:
            dt = datetime.fromisoformat(item.createdAt.replace('Z', '+00:00'))
        except: dt = datetime.now(timezone.utc)
        
        activities.append(UnifiedActivity(
            source_id=item.id, source_platform="linear",
            type=ActivityType.TICKET, author="LinearUser",
            content=f"{item.title}\n{item.description or ''}",
            timestamp=dt, url=item.url, files_changed=[]
        ))

    logger.info(f"✅ Ingestion complete. Total items: {len(activities)}")
    return {"activities": activities}

# --- NODE 2: DB WRITER (PERSISTENCE) ---
def db_writer_node(state: GraphState):
    report = state["final_report"]
    repo_name = state["repo_name"]
    
    if not report: return {}

    logger.info("💾 Persisting Analysis Report to SQL...")
    session_gen = get_session()
    session = next(session_gen)
    
    try:
        # 1. Ensure Repo Exists
        repo_db = session.exec(select(Repository).where(Repository.name == repo_name)).first()
        if not repo_db:
            repo_db = Repository(url=f"https://github.com/{repo_name}", name=repo_name)
            session.add(repo_db)
            session.commit()
            session.refresh(repo_db)

        # 2. Save Report
        # IMPORTANTE: No usamos strings para relationships si podemos evitarlo para simplificar el registry
        db_report = AnalysisReport(
            repository_id=repo_db.id,
            developer_name=report.developer_name,
            quality_score=report.quality_score,
            prs_merged=report.prs_merged,
            feedback_summary=report.feedback_summary
        )
        session.add(db_report)
        session.commit()
        logger.info(f"✅ Report saved with ID: {db_report.id}")
        
    except Exception as e:
        logger.error(f"❌ DB Write Error: {e}")
    finally:
        session.close()

    return {}

# --- GRAPH ASSEMBLY ---
def build_analyst_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("ingest", ingest_data_node)
    workflow.add_node("analyst", analyze_activities_node) 
    workflow.add_node("reporter", generate_report_node)   
    workflow.add_node("writer", db_writer_node)

    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "analyst")
    workflow.add_edge("analyst", "reporter")
    workflow.add_edge("reporter", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()