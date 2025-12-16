# app/core/agents/analyst_graph.py
import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Optional
from langgraph.graph import StateGraph, END
from app.core.agents.state import GraphState
from app.core.agents.nodes import analyze_activities_node, generate_report_node
from app.adapters.github.adapter import GitHubAdapter
from app.adapters.linear.adapter import LinearAdapter
from app.core.models.domain import UnifiedActivity, ActivityType
from app.core.database.session import engine, get_session
from app.core.database.models import (
    Activity,
    AnalysisReport,
    IntegrationCredential,
    Project,
    ProjectOwner,
    ProjectIntegration,
    ProjectRepository,
    Repository,
    Ticket,
    TicketEvent,
)
from app.core.security.crypto import decrypt_secret
from sqlmodel import Session, select
from app.core.logger import get_logger

logger = get_logger(__name__)

# --- NODE 1: PARALLEL INGESTION ---
async def ingest_data_node(state: GraphState):
    """
    Fetches data from GitHub and Linear simultaneously using asyncio.
    """
    repo_name = state["repo_name"]
    project_name = state.get("project_name") or repo_name
    # Use lookback_days from state (default 90).
    days = state.get("lookback_days", 90)
    linear_team_key = state.get("linear_team_key")
    user_id = state.get("user_id")
    
    logger.info(f"🚀 Starting Parallel Ingestion for: {repo_name} (Lookback: {days} days)")

    github_token = None
    linear_api_key = None
    if user_id:
        try:
            with Session(engine) as session:
                github_cred = session.exec(
                    select(IntegrationCredential)
                    .where(
                        IntegrationCredential.owner_id == user_id,
                        IntegrationCredential.provider == "github",
                    )
                    .order_by(IntegrationCredential.updated_at.desc())
                ).first()
                if github_cred:
                    github_token = decrypt_secret(github_cred.encrypted_secret)

                project_db = session.exec(
                    select(Project)
                    .join(ProjectOwner, ProjectOwner.project_id == Project.id)
                    .where(Project.name == project_name)
                    .where(ProjectOwner.owner_id == user_id)
                ).first()
                if project_db:
                    linear_integration = session.exec(
                        select(ProjectIntegration)
                        .where(
                            ProjectIntegration.project_id == project_db.id,
                            ProjectIntegration.provider == "linear",
                        )
                        .order_by(ProjectIntegration.updated_at.desc())
                    ).first()

                    if linear_integration:
                        if not linear_team_key:
                            linear_team_key = linear_integration.settings.get("team_key")

                        if linear_integration.credential_id:
                            linear_cred = session.get(IntegrationCredential, linear_integration.credential_id)
                            if linear_cred:
                                linear_api_key = decrypt_secret(linear_cred.encrypted_secret)
        except Exception as e:
            logger.warning(f"Failed to load integration credentials from DB: {e}")

    gh_adapter = GitHubAdapter(token=github_token)
    
    # Determine if we should run Linear ingestion
    # If user_id is present (production/multi-tenant), we MUST have found a key in DB.
    # If user_id is missing (local dev), we allow fallback to env vars inside Adapter.
    should_run_linear = True
    if user_id and not linear_api_key:
        should_run_linear = False
        logger.info("Skipping Linear Sync: No Linear credential found for project.")

    # Create async tasks (wrap synchronous calls in threads)
    tasks = [asyncio.to_thread(gh_adapter.fetch_recent_activity, repo_name, days)]
    
    if should_run_linear:
        lin_adapter = LinearAdapter(api_key=linear_api_key)
        tasks.append(asyncio.to_thread(lin_adapter.fetch_recent_issues, 200, team_key=linear_team_key))
    
    # Execute in parallel
    results = await asyncio.gather(*tasks)
    gh_data = results[0]
    lin_data = results[1] if len(results) > 1 else []

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
        def _parse_dt(value: Optional[str]) -> Optional[datetime]:
            if not value:
                return None
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:
                return None

        created_at = _parse_dt(getattr(item, "createdAt", None))
        updated_at = _parse_dt(getattr(item, "updatedAt", None)) or created_at
        completed_at = _parse_dt(getattr(item, "completedAt", None))
        dt = updated_at or created_at or datetime.now(timezone.utc)

        cycle_starts_at = _parse_dt(getattr(item, "cycle_startsAt", None))
        cycle_ends_at = _parse_dt(getattr(item, "cycle_endsAt", None))
        
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
            created_at=created_at,
            updated_at=updated_at,
            completed_at=completed_at,
            assignee=getattr(item, "assignee", None),
            status_type=getattr(item, "state_type", None),
            cycle_name=getattr(item, "cycle_name", None),
            cycle_number=getattr(item, "cycle_number", None),
            cycle_starts_at=cycle_starts_at,
            cycle_ends_at=cycle_ends_at,
        ))

    logger.info(f"✅ Ingestion complete. Total items: {len(activities)}")
    return {"activities": activities}

# --- NODE 2: DB WRITER (PERSISTENCE) ---
def db_writer_node(state: GraphState):
    report = state["final_report"]
    repo_name = state["repo_name"]
    project_name = state.get("project_name") or repo_name
    activities = state.get("activities", [])
    owner_id = state.get("user_id")
    
    if not report: return {}
    if not owner_id:
        logger.error("Missing user_id in graph state; cannot persist multi-tenant data safely.")
        return {}

    logger.info("💾 Persisting Analysis Report to SQL...")
    session_gen = get_session()
    session = next(session_gen)
    
    try:
        # 1. Ensure Repository Exists
        repo_db = session.exec(
            select(Repository)
            .where(Repository.owner_id == owner_id)
            .where(Repository.name == repo_name)
        ).first()
        if not repo_db:
            repo_db = Repository(
                url=f"https://github.com/{repo_name}",
                name=repo_name,
                owner_id=owner_id,
            )
            session.add(repo_db)
            session.commit()
            session.refresh(repo_db)

        # Update last analyzed timestamp
        repo_db.last_analyzed = datetime.utcnow()

        # 2. Ensure Project Exists and link Repository <-> Project
        project_db = session.exec(
            select(Project)
            .join(ProjectOwner, ProjectOwner.project_id == Project.id)
            .where(Project.name == project_name)
            .where(ProjectOwner.owner_id == owner_id)
        ).first()
        if not project_db:
            existing_by_name = session.exec(select(Project).where(Project.name == project_name)).first()
            if existing_by_name:
                owner_row = session.get(ProjectOwner, existing_by_name.id)
                if owner_row and owner_row.owner_id != owner_id:
                    logger.error("Project name already exists for a different user; cannot persist analysis safely.")
                    return {}
                project_db = existing_by_name
            else:
                project_db = Project(name=project_name)
                session.add(project_db)
                session.commit()
                session.refresh(project_db)

            if not session.get(ProjectOwner, project_db.id):
                session.add(ProjectOwner(project_id=project_db.id, owner_id=owner_id))
                session.commit()

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
            incoming_assignee = getattr(act, "assignee", None) or getattr(act, "author", None)
            incoming_status_type = getattr(act, "status_type", None)
            incoming_created_at = getattr(act, "created_at", None) or act.timestamp
            incoming_updated_at = getattr(act, "updated_at", None) or act.timestamp
            incoming_completed_at = getattr(act, "completed_at", None)

            fingerprint_payload = "|".join(
                [
                    str(incoming_key or ""),
                    str(act.title or ""),
                    str(act.content or ""),
                    str(incoming_points),
                    str(incoming_status or ""),
                    str(incoming_status_type or ""),
                    str(incoming_assignee or ""),
                    str(incoming_updated_at or ""),
                    str(incoming_completed_at or ""),
                ]
            )
            incoming_fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()

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
                    created_at=incoming_created_at,
                    updated_at=incoming_updated_at,
                    completed_at=incoming_completed_at,
                    url=act.url,
                    raw_payload={
                        "external_key": incoming_key,
                        "status_label": incoming_status,
                        "story_points": incoming_points,
                        "assignee": incoming_assignee,
                        "status_type": incoming_status_type,
                        "cycle_name": getattr(act, "cycle_name", None),
                        "cycle_number": getattr(act, "cycle_number", None),
                        "cycle_startsAt": (
                            getattr(act, "cycle_starts_at", None).isoformat()
                            if getattr(act, "cycle_starts_at", None)
                            else None
                        ),
                        "cycle_endsAt": (
                            getattr(act, "cycle_ends_at", None).isoformat()
                            if getattr(act, "cycle_ends_at", None)
                            else None
                        ),
                        "fingerprint": incoming_fingerprint,
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

            stored_fingerprint = (ticket_db.raw_payload or {}).get("fingerprint")
            if stored_fingerprint and stored_fingerprint == incoming_fingerprint:
                continue

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

            current_assignee = (ticket_db.raw_payload or {}).get("assignee")
            if incoming_assignee != current_assignee:
                session.add(
                    TicketEvent(
                        ticket_id=ticket_db.id,
                        event_type="assignee_changed",
                        from_value=str(current_assignee),
                        to_value=str(incoming_assignee),
                    )
                )

            if incoming_completed_at and ticket_db.completed_at != incoming_completed_at:
                session.add(
                    TicketEvent(
                        ticket_id=ticket_db.id,
                        event_type="completed_at_changed",
                        from_value=str(ticket_db.completed_at),
                        to_value=str(incoming_completed_at),
                    )
                )

            ticket_db.description = act.content or ticket_db.description
            ticket_db.updated_at = incoming_updated_at
            ticket_db.completed_at = incoming_completed_at or ticket_db.completed_at
            ticket_db.url = act.url or ticket_db.url
            ticket_db.raw_payload = {
                "external_key": ticket_db.key,
                "status_label": ticket_db.status_label,
                "story_points": ticket_db.story_points,
                "assignee": incoming_assignee,
                "status_type": incoming_status_type,
                "cycle_name": getattr(act, "cycle_name", None),
                "cycle_number": getattr(act, "cycle_number", None),
                "cycle_startsAt": (
                    getattr(act, "cycle_starts_at", None).isoformat()
                    if getattr(act, "cycle_starts_at", None)
                    else None
                ),
                "cycle_endsAt": (
                    getattr(act, "cycle_ends_at", None).isoformat()
                    if getattr(act, "cycle_ends_at", None)
                    else None
                ),
                "fingerprint": incoming_fingerprint,
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
            security_alerts=report.security_alerts,
            risk_details=report.risk_details,
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
