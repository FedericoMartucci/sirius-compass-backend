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
from app.core.database.models import (
    Activity,
    AnalysisReport,
    Project,
    ProjectRepository,
    Repository,
    Ticket,
    TicketEvent,
)
from sqlmodel import select
from app.core.logger import get_logger

logger = get_logger(__name__)

# --- NODE 1: PARALLEL INGESTION ---
async def ingest_data_node(state: GraphState):
    """
    Fetches data from GitHub and Linear simultaneously using asyncio.
    """
    repo_name = state["repo_name"]
    # Use lookback_days from state (default 90).
    days = state.get("lookback_days", 90)
    linear_team_key = state.get("linear_team_key")
    
    logger.info(f"🚀 Starting Parallel Ingestion for: {repo_name} (Lookback: {days} days)")

    gh_adapter = GitHubAdapter()
    lin_adapter = LinearAdapter()

    # Create async tasks (wrap synchronous calls in threads)
    task_gh = asyncio.to_thread(gh_adapter.fetch_recent_activity, repo_name, days)
    task_lin = asyncio.to_thread(lin_adapter.fetch_recent_issues, 200, team_key=linear_team_key)

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
            title=item.content.splitlines()[0].strip() if item.content else None,
            content=item.content, timestamp=item.timestamp,
            url=f"https://github.com/{repo_name}/commit/{item.source_id}",
            files_changed=item.files_changed
        ))

    # Map Linear
    for item in lin_data:
        try:
            dt_source = getattr(item, "updatedAt", None) or item.createdAt
            dt = datetime.fromisoformat(dt_source.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.now(timezone.utc)
        
        activities.append(UnifiedActivity(
            source_id=item.id, source_platform="linear",
            type=ActivityType.TICKET, author=item.assignee or "Unassigned",
            title=item.title,
            content=f"[{item.identifier}] {item.title}\n{item.description or ''}",
            timestamp=dt,
            url=item.url,
            files_changed=[],
            story_points=int(getattr(item, "estimate", 0) or 0),
            status_label=getattr(item, "state", None),
            external_key=getattr(item, "identifier", None),
        ))

    logger.info(f"✅ Ingestion complete. Total items: {len(activities)}")
    return {"activities": activities}

# --- NODE 2: DB WRITER (PERSISTENCE) ---
def db_writer_node(state: GraphState):
    report = state["final_report"]
    repo_name = state["repo_name"]
    project_name = state.get("project_name") or repo_name
    activities = state.get("activities", [])
    
    if not report: return {}

    logger.info("💾 Persisting Analysis Report to SQL...")
    session_gen = get_session()
    session = next(session_gen)
    
    try:
        # 1. Ensure Repository Exists
        repo_db = session.exec(select(Repository).where(Repository.name == repo_name)).first()
        if not repo_db:
            repo_db = Repository(url=f"https://github.com/{repo_name}", name=repo_name)
            session.add(repo_db)
            session.commit()
            session.refresh(repo_db)

        # Update last analyzed timestamp
        repo_db.last_analyzed = datetime.utcnow()

        # 2. Ensure Project Exists and link Repository <-> Project
        project_db = session.exec(select(Project).where(Project.name == project_name)).first()
        if not project_db:
            project_db = Project(name=project_name)
            session.add(project_db)
            session.commit()
            session.refresh(project_db)

        project_repo_link = session.exec(
            select(ProjectRepository).where(
                ProjectRepository.project_id == project_db.id,
                ProjectRepository.repository_id == repo_db.id,
            )
        ).first()
        if not project_repo_link:
            session.add(
                ProjectRepository(
                    project_id=project_db.id,
                    repository_id=repo_db.id,
                    is_primary=True,
                )
            )

        # 3. Persist code activities (GitHub only) to avoid incorrectly binding tickets to a single repo.
        latest_code_timestamp = None
        for act in activities:
            if act.source_platform != "github":
                continue

            exists = session.exec(
                select(Activity).where(
                    Activity.repository_id == repo_db.id,
                    Activity.source_id == act.source_id,
                )
            ).first()
            if exists:
                continue

            title = act.title or (act.content.splitlines()[0].strip() if act.content else act.type.value)
            session.add(
                Activity(
                    repository_id=repo_db.id,
                    source_platform=act.source_platform,
                    source_id=act.source_id,
                    type=act.type.value if hasattr(act.type, "value") else str(act.type),
                    author=act.author,
                    timestamp=act.timestamp,
                    title=title[:255],
                    content=act.content,
                    files_changed_count=len(act.files_changed or []),
                )
            )

            if latest_code_timestamp is None or act.timestamp > latest_code_timestamp:
                latest_code_timestamp = act.timestamp

        if latest_code_timestamp is not None:
            repo_db.last_synced_at = latest_code_timestamp

        # 4. Persist tickets (Linear) into Ticket and generate TicketEvent by diffing snapshots.
        for act in activities:
            type_label = act.type.value if hasattr(act.type, "value") else str(act.type)
            if act.source_platform != "linear" or type_label != "TICKET":
                continue

            incoming_points = int(getattr(act, "story_points", 0) or 0)
            incoming_status = getattr(act, "status_label", None)
            incoming_key = getattr(act, "external_key", None)

            ticket_db = session.exec(select(Ticket).where(Ticket.source_id == act.source_id)).first()
            if not ticket_db:
                ticket_db = Ticket(
                    project_id=project_db.id,
                    source_platform="linear",
                    source_id=act.source_id,
                    key=incoming_key,
                    title=act.title or "Untitled",
                    description=act.content,
                    story_points=incoming_points,
                    status_label=incoming_status,
                    created_at=act.timestamp,
                    updated_at=datetime.utcnow(),
                    url=act.url,
                    raw_payload={
                        "external_key": incoming_key,
                        "status_label": incoming_status,
                        "story_points": incoming_points,
                    },
                )
                session.add(ticket_db)
                session.flush()
                session.add(
                    TicketEvent(
                        ticket_id=ticket_db.id,
                        event_type="created",
                        to_value=incoming_status,
                        payload={"key": incoming_key},
                    )
                )
                continue

            if ticket_db.project_id != project_db.id:
                ticket_db.project_id = project_db.id

            if incoming_key and incoming_key != ticket_db.key:
                session.add(
                    TicketEvent(
                        ticket_id=ticket_db.id,
                        event_type="key_changed",
                        from_value=ticket_db.key,
                        to_value=incoming_key,
                    )
                )
                ticket_db.key = incoming_key

            if incoming_status != ticket_db.status_label:
                session.add(
                    TicketEvent(
                        ticket_id=ticket_db.id,
                        event_type="status_changed",
                        from_value=ticket_db.status_label,
                        to_value=incoming_status,
                    )
                )
                ticket_db.status_label = incoming_status

            if incoming_points != ticket_db.story_points:
                session.add(
                    TicketEvent(
                        ticket_id=ticket_db.id,
                        event_type="points_changed",
                        from_value=str(ticket_db.story_points),
                        to_value=str(incoming_points),
                    )
                )
                ticket_db.story_points = incoming_points

            if act.title and act.title != ticket_db.title:
                session.add(
                    TicketEvent(
                        ticket_id=ticket_db.id,
                        event_type="title_changed",
                        from_value=ticket_db.title,
                        to_value=act.title,
                    )
                )
                ticket_db.title = act.title

            ticket_db.description = act.content or ticket_db.description
            ticket_db.updated_at = datetime.utcnow()
            ticket_db.url = act.url or ticket_db.url
            ticket_db.raw_payload = {
                "external_key": ticket_db.key,
                "status_label": ticket_db.status_label,
                "story_points": ticket_db.story_points,
            }

        # Commit ingestion data before attempting to write the report.
        session.commit()

        # 5. Save Report
        commit_like_types = {"COMMIT", "PR_MERGE"}
        commit_count = 0
        for a in activities:
            type_label = a.type.value if hasattr(a.type, "value") else str(a.type)
            if type_label in commit_like_types or type_label.endswith("COMMIT") or type_label.endswith("PR_MERGE"):
                commit_count += 1

        db_report = AnalysisReport(
            repository_id=repo_db.id,
            developer_name=report.developer_name,
            quality_score=report.quality_score,
            prs_merged=report.prs_merged,
            commits_count=commit_count,
            detected_skills=getattr(report, "detected_skills", []) or [],
            security_alerts=False,
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
