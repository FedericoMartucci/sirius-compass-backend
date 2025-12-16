# app/api/server.py
import asyncio
import os
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Request, status
from fastapi.responses import StreamingResponse
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Depends, Query, Body
# from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from sqlmodel import Session, select
from sqlalchemy import false

# Import Schemas from your existing file
from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatMessageDTO,
    ChatRequest,
    ChatThreadDTO,
    ConnectionDTO,
    CreateConnectionRequest,
    GuestDTO,
    InviteGuestRequest,
    ProjectGuestDTO,
    CreateProjectRequest,
    ProjectDTO,
    ReportDTO,
    SyncRequest,
    SyncRunDTO,
    SaveUserSettingsPayload,
    UserSettingsDTO,
)

from app.core.database.session import create_db_and_tables, engine
from app.core.agents.analyst_graph import build_analyst_graph
from app.core.agents.chat_graph import build_chat_graph
from app.core.logger import get_logger
from app.core.security.crypto import encrypt_secret
from app.core.security.auth import get_current_user, get_user_id
from app.core.streaming import TokenStreamHandler, sse_data
from app.core.database.models import (
    AnalysisReport,
    ChatMessage,
    ChatThread,
    DataCoverage,
    Guest,
    IntegrationCredential,
    Project,
    ProjectGuest,
    ProjectOwner,
    ProjectIntegration,
    ProjectRepository,
    Repository,
    SyncRun,
    Ticket,
    TicketEvent,
    UserSettings,
)
from app.services.chat_storage import (
    append_message,
    coerce_content_to_text,
    get_or_create_thread,
    load_thread_messages,
)
from app.services.sync_queue import enqueue_sync_run

load_dotenv()
logger = get_logger(__name__)


ALLOWED_TIME_RANGES = {"7d", "30d", "90d", "180d", "365d", "all"}
DEFAULT_TIME_RANGE = "30d"


def _get_user_email(user: dict) -> Optional[str]:
    email = user.get("email")
    if isinstance(email, str):
        email = email.strip().lower()
    return email if email else None


def _ensure_guest_claim(session: Session, *, user_id: str, email: Optional[str]) -> None:
    """Best-effort: if the authenticated user has an email, link it to Guest.external_user_id."""
    if not email:
        return
    guest = session.exec(select(Guest).where(Guest.email == email)).first()
    if not guest:
        return
    if guest.external_user_id and guest.external_user_id != user_id:
        return
    if guest.external_user_id != user_id:
        guest.external_user_id = user_id
        if guest.accepted_at is None:
            guest.accepted_at = datetime.utcnow()
        session.add(guest)
        session.commit()

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


def _format_relative_time(dt: Optional[datetime]) -> str:
    if not dt:
        return "Never"

    now = datetime.now(timezone.utc)
    dt_utc = dt
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    delta = now - dt_utc
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "Just now"

    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minutes ago"

    hours = minutes // 60
    if hours < 24:
        return f"{hours} hours ago"

    days = hours // 24
    return f"{days} days ago"

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
async def analyze_repo(
    request: AnalyzeRequest,
    user_id: str = Depends(get_user_id)
):
    """
    Triggers the heavy analysis graph (GitHub + Linear -> Report -> DB).
    Requires authentication - scoped to authenticated user.
    """
    logger.info(f"🚀 Starting Analysis for {request.repo_url} (Lookback: {request.lookback_days} days) [User: {user_id}]")
    
    try:
        # 1. Parse Repo Name
        repo_name = _parse_repo_name(request.repo_url)
        project_name = request.project_name or repo_name
        
        # Prevent heavy analysis while a sync is running for the same repo.
        with Session(engine) as session:
            repo_row = session.exec(
                select(Repository)
                .where(Repository.owner_id == user_id)
                .where(Repository.name == repo_name)
            ).first()
            if repo_row:
                active = session.exec(
                    select(SyncRun)
                    .where(SyncRun.owner_id == user_id)
                    .where(SyncRun.repository_id == repo_row.id)
                    .where(SyncRun.status.in_(["queued", "running"]))
                    .order_by(SyncRun.created_at.desc())
                ).first()
                if active:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Sync in progress (run_id={active.id}). Please wait until it finishes before running analysis.",
                    )

        # 2. Build Graph
        workflow = build_analyst_graph()
        
        # 3. Initial State (use authenticated user_id)
        initial_state = {
            "repo_name": repo_name,
            "project_name": project_name,
            "developer_name": request.developer_name,
            "lookback_days": request.lookback_days,
            "linear_team_key": request.linear_team_key,
            "user_id": user_id,
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
async def chat_agent(
    payload: ChatRequest,
    http_request: Request,
    user_id: str = Depends(get_user_id)
):
    """
    Talks to the Conversational Graph.
    Requires authentication - scoped to authenticated user's threads.
    """
    try:
        project_name = payload.project_name or payload.repo_name
        with Session(engine) as session:
            # Always use authenticated user_id for thread ownership
            try:
                thread = get_or_create_thread(
                    session,
                    external_thread_id=payload.thread_id,
                    owner_id=user_id,
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=str(e),
                ) from e
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
                        {
                            "messages": messages,
                            "meta": {
                                "user_id": user_id,
                                "thread_id": payload.thread_id,
                                "project_name": project_name,
                                "repo_name": payload.repo_name,
                            },
                        },
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

        result = await workflow.ainvoke(
            {
                "messages": messages,
                "meta": {
                    "user_id": user_id,
                    "thread_id": payload.thread_id,
                    "project_name": project_name,
                    "repo_name": payload.repo_name,
                },
            }
        )
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


# --- Endpoint 3: PROJECTS ---
@app.get("/projects", response_model=List[ProjectDTO])
def list_projects(user: dict = Depends(get_current_user)):
    """List projects accessible to the authenticated user (owner or guest)."""
    user_id = get_user_id(user)
    email = _get_user_email(user)

    with Session(engine) as session:
        _ensure_guest_claim(session, user_id=user_id, email=email)

        # Backfill ownership for legacy data:
        # - projects linked to repos owned by this user
        # - projects linked to integrations whose credentials are owned by this user
        legacy_repo_projects = session.exec(
            select(Project)
            .join(ProjectRepository, Project.id == ProjectRepository.project_id)
            .join(Repository, ProjectRepository.repository_id == Repository.id)
            .outerjoin(ProjectOwner, ProjectOwner.project_id == Project.id)
            .where(Repository.owner_id == user_id)
            .where(ProjectOwner.project_id == None)  # noqa: E711
        ).all()

        legacy_linear_projects = session.exec(
            select(Project)
            .join(ProjectIntegration, ProjectIntegration.project_id == Project.id)
            .join(IntegrationCredential, IntegrationCredential.id == ProjectIntegration.credential_id)
            .outerjoin(ProjectOwner, ProjectOwner.project_id == Project.id)
            .where(IntegrationCredential.owner_id == user_id)
            .where(ProjectOwner.project_id == None)  # noqa: E711
        ).all()

        to_claim = {p.id for p in (legacy_repo_projects + legacy_linear_projects) if p.id is not None}
        if to_claim:
            for project_id in to_claim:
                if not session.get(ProjectOwner, project_id):
                    session.add(ProjectOwner(project_id=project_id, owner_id=user_id))
            session.commit()

        projects = session.exec(
            select(Project)
            .outerjoin(ProjectOwner, ProjectOwner.project_id == Project.id)
            .outerjoin(ProjectGuest, ProjectGuest.project_id == Project.id)
            .outerjoin(Guest, Guest.id == ProjectGuest.guest_id)
            .where(
                (ProjectOwner.owner_id == user_id)
                | (Guest.external_user_id == user_id)
                | ((Guest.email == email) if email is not None else false())
            )
            .distinct()
            .order_by(Project.updated_at.desc())
        ).all()
        return [ProjectDTO(id=str(p.id), name=p.name) for p in projects]


@app.post("/projects", response_model=ProjectDTO)
def create_project(
    payload: CreateProjectRequest,
    user_id: str = Depends(get_user_id)
):
    """Create a new project for the authenticated user."""
    name = payload.name.strip()
    with Session(engine) as session:
        existing = session.exec(
            select(Project)
            .join(ProjectOwner, ProjectOwner.project_id == Project.id)
            .where(Project.name == name)
            .where(ProjectOwner.owner_id == user_id)
        ).first()
        if existing:
            return ProjectDTO(id=str(existing.id), name=existing.name)

        # Project names are globally unique in the current schema. If a project already exists,
        # we only allow using it if it's unclaimed or already owned by this user.
        project = session.exec(select(Project).where(Project.name == name)).first()
        if project:
            owner_row = session.get(ProjectOwner, project.id)
            if owner_row and owner_row.owner_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A project with this name already exists for a different user.",
                )
            if not owner_row:
                session.add(ProjectOwner(project_id=project.id, owner_id=user_id))
                session.commit()
            return ProjectDTO(id=str(project.id), name=project.name)

        project = Project(name=name, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        session.add(project)
        session.commit()
        session.refresh(project)
        session.add(ProjectOwner(project_id=project.id, owner_id=user_id))
        session.commit()
        return ProjectDTO(id=str(project.id), name=project.name)


# --- Endpoint 3.5: USER SETTINGS ---
def _coerce_project_id(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="default_project_id must be a stringified integer or null")


def _normalize_time_range(value: Optional[str]) -> str:
    if value is None:
        return DEFAULT_TIME_RANGE
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="default_time_range must be a string")
    value = value.strip()
    if not value:
        return DEFAULT_TIME_RANGE
    if value not in ALLOWED_TIME_RANGES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid default_time_range",
                "allowed": sorted(ALLOWED_TIME_RANGES),
            },
        )
    return value


def _project_accessible(
    session: Session,
    *,
    project_id: int,
    user_id: str,
    email: Optional[str],
) -> bool:
    row = session.exec(
        select(Project.id)
        .outerjoin(ProjectOwner, ProjectOwner.project_id == Project.id)
        .outerjoin(ProjectGuest, ProjectGuest.project_id == Project.id)
        .outerjoin(Guest, Guest.id == ProjectGuest.guest_id)
        .where(Project.id == project_id)
        .where(
            (ProjectOwner.owner_id == user_id)
            | (Guest.external_user_id == user_id)
            | ((Guest.email == email) if email is not None else false())
        )
        .limit(1)
    ).first()
    return row is not None


@app.get("/user-settings", response_model=UserSettingsDTO)
def get_user_settings(user: dict = Depends(get_current_user)):
    """Return authenticated user's settings (or defaults if none exist)."""
    user_id = get_user_id(user)
    email = _get_user_email(user)

    with Session(engine) as session:
        settings = session.exec(select(UserSettings).where(UserSettings.user_id == user_id)).first()
        if not settings:
            return UserSettingsDTO(default_project_id=None, default_time_range=DEFAULT_TIME_RANGE)

        # If stored project no longer exists / is not accessible, return null (and clean up).
        default_project_id: Optional[str] = None
        if settings.default_project_id is not None:
            if _project_accessible(
                session,
                project_id=settings.default_project_id,
                user_id=user_id,
                email=email,
            ):
                default_project_id = str(settings.default_project_id)
            else:
                settings.default_project_id = None
                settings.updated_at = datetime.utcnow()
                session.add(settings)
                session.commit()

        return UserSettingsDTO(
            default_project_id=default_project_id,
            default_time_range=settings.default_time_range or DEFAULT_TIME_RANGE,
        )


@app.put("/user-settings", response_model=UserSettingsDTO)
def save_user_settings(
    payload: SaveUserSettingsPayload,
    user: dict = Depends(get_current_user),
):
    """Upsert authenticated user's settings."""
    user_id = get_user_id(user)
    email = _get_user_email(user)

    requested_project_id = _coerce_project_id(payload.default_project_id)
    requested_time_range = payload.default_time_range

    with Session(engine) as session:
        settings = session.exec(select(UserSettings).where(UserSettings.user_id == user_id)).first()
        if not settings:
            settings = UserSettings(
                user_id=user_id,
                default_project_id=None,
                default_time_range=DEFAULT_TIME_RANGE,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

        if requested_project_id is not None:
            if not _project_accessible(
                session,
                project_id=requested_project_id,
                user_id=user_id,
                email=email,
            ):
                raise HTTPException(status_code=400, detail="default_project_id is not accessible for this user")
            settings.default_project_id = requested_project_id
        elif payload.default_project_id is not None:
            # Explicit null/empty clears.
            settings.default_project_id = None

        if requested_time_range is not None:
            settings.default_time_range = _normalize_time_range(requested_time_range)

        settings.updated_at = datetime.utcnow()
        session.add(settings)
        session.commit()
        session.refresh(settings)

        return UserSettingsDTO(
            default_project_id=str(settings.default_project_id) if settings.default_project_id is not None else None,
            default_time_range=settings.default_time_range or DEFAULT_TIME_RANGE,
        )


# --- Endpoint 4: CONNECTIONS (Integrations) ---
@app.get("/connections", response_model=List[ConnectionDTO])
def list_connections(
    project_name: Optional[str] = Query(default=None),
    user: dict = Depends(get_current_user)
):
    """List connections scoped to authenticated user's projects."""
    user_id = get_user_id(user)
    email = _get_user_email(user)
    with Session(engine) as session:
        _ensure_guest_claim(session, user_id=user_id, email=email)

        def get_status_from_run(run: Optional[SyncRun]) -> tuple[str, Optional[str]]:
            if not run:
                return "active", None
            if run.status in {"queued", "running"}:
                return "syncing", None
            if run.status == "failed":
                return "error", run.message
            return "active", None

        projects: List[Project] = []
        # Fetch projects accessible to this user (owner or guest), even if they have no repositories yet.
        base_query = (
            select(Project)
            .outerjoin(ProjectOwner, ProjectOwner.project_id == Project.id)
            .outerjoin(ProjectGuest, ProjectGuest.project_id == Project.id)
            .outerjoin(Guest, Guest.id == ProjectGuest.guest_id)
            .where(
                (ProjectOwner.owner_id == user_id)
                | (Guest.external_user_id == user_id)
                | ((Guest.email == email) if email is not None else false())
            )
            .distinct()
        )
        
        if project_name:
            project = session.exec(base_query.where(Project.name == project_name)).first()
            if project:
                projects = [project]
        else:
            projects = session.exec(base_query).all()

        connections: List[ConnectionDTO] = []

        project_ids = [p.id for p in projects if p.id is not None]
        repo_links: list[ProjectRepository] = []
        repo_ids: list[int] = []
        if project_ids:
            repo_links = session.exec(
                select(ProjectRepository).where(ProjectRepository.project_id.in_(project_ids))
            ).all()
            repo_ids = [l.repository_id for l in repo_links if l.repository_id is not None]

        # Fetch latest sync runs for these scopes (regardless of who owns the run).
        # This makes connection status useful for guests too.
        latest_repo_run: dict[int, SyncRun] = {}
        latest_project_run: dict[int, SyncRun] = {}
        if project_ids or repo_ids:
            all_recent_runs = session.exec(
                select(SyncRun)
                .where(
                    (SyncRun.project_id.in_(project_ids) if project_ids else false())
                    | (SyncRun.repository_id.in_(repo_ids) if repo_ids else false())
                )
                .order_by(SyncRun.created_at.desc())
                .limit(500)
            ).all()

            for r in all_recent_runs:
                if r.repository_id and r.repository_id not in latest_repo_run:
                    latest_repo_run[r.repository_id] = r
                if r.project_id and r.project_id not in latest_project_run and r.provider in {"linear", "all"}:
                    latest_project_run[r.project_id] = r

        # Repository connections
        for project in projects:
            links = [l for l in repo_links if l.project_id == project.id]
            for link in links:
                repo = session.get(Repository, link.repository_id)
                if not repo:
                    continue

                run = latest_repo_run.get(repo.id or 0)
                status, error = get_status_from_run(run)
                
                # Use run timestamp if available (user preference), else fallback to repo field
                last_sync_time = run.created_at if run else repo.last_synced_at

                connections.append(
                    ConnectionDTO(
                        id=repo.id or 0,
                        type="Repository",
                        name=repo.name,
                        project=project.name,
                        status=status,
                        lastSync=_format_relative_time(last_sync_time),
                        last_error=error,
                    )
                )

        # Ticket board connections (Linear)
        for project in projects:
            integration = session.exec(
                select(ProjectIntegration).where(
                    ProjectIntegration.project_id == project.id,
                    ProjectIntegration.provider == "linear",
                )
            ).first()

            has_tickets = session.exec(
                select(Ticket.id).where(
                    Ticket.project_id == project.id,
                    Ticket.source_platform == "linear",
                )
            ).first() is not None

            if not integration and not has_tickets:
                continue

            team_key = (integration.settings.get("team_key") if integration else None) or "unknown"
            last_ticket = session.exec(
                select(Ticket)
                .where(
                    Ticket.project_id == project.id,
                    Ticket.source_platform == "linear",
                )
                .order_by(Ticket.updated_at.desc())
            ).first()

            last_sync_dt = None
            if last_ticket and last_ticket.updated_at:
                last_sync_dt = last_ticket.updated_at
            elif integration:
                last_sync_dt = integration.updated_at
            
            run = latest_project_run.get(project.id or 0)
            status, error = get_status_from_run(run)

            # Use run timestamp if available (user preference)
            if run:
                last_sync_dt = run.created_at

            connections.append(
                ConnectionDTO(
                    id=(integration.id if integration and integration.id else 1_000_000 + (project.id or 0)),
                    type="Board",
                    name=f"Linear ({team_key})",
                    project=project.name,
                    status=status,
                    lastSync=_format_relative_time(last_sync_dt),
                    last_error=error,
                )
            )

        return connections


@app.post("/projects/guests", response_model=ProjectGuestDTO)
def invite_guest_to_project(
    payload: InviteGuestRequest,
    user: dict = Depends(get_current_user),
):
    """Invite a guest (by email) to a project. Only the project owner can invite."""
    user_id = get_user_id(user)
    email = (payload.email or "").strip().lower()
    role = (payload.role or "viewer").strip().lower()
    if role not in {"viewer", "editor"}:
        raise HTTPException(status_code=400, detail="role must be 'viewer' or 'editor'")
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")

    with Session(engine) as session:
        project: Optional[Project] = None
        if payload.project_id is not None:
            project = session.get(Project, payload.project_id)
        elif payload.project_name:
            project = session.exec(select(Project).where(Project.name == payload.project_name.strip())).first()
        else:
            raise HTTPException(status_code=400, detail="project_id or project_name is required")

        if not project or project.id is None:
            raise HTTPException(status_code=404, detail="Project not found")

        owner_row = session.get(ProjectOwner, project.id)
        if not owner_row or owner_row.owner_id != user_id:
            raise HTTPException(status_code=403, detail="Only the project owner can invite guests")

        guest = session.exec(select(Guest).where(Guest.email == email)).first()
        if not guest:
            guest = Guest(email=email, invited_by_owner_id=user_id)
            session.add(guest)
            session.commit()
            session.refresh(guest)

        link = session.exec(
            select(ProjectGuest)
            .where(ProjectGuest.project_id == project.id)
            .where(ProjectGuest.guest_id == guest.id)
        ).first()
        if not link:
            link = ProjectGuest(project_id=project.id, guest_id=guest.id or 0, role=role)
        else:
            link.role = role
        session.add(link)
        session.commit()
        session.refresh(link)

        return ProjectGuestDTO(
            project_id=project.id,
            guest=GuestDTO(id=guest.id or 0, email=guest.email),
            role=link.role,
            created_at=link.created_at,
        )


@app.get("/projects/{project_id}/guests", response_model=List[ProjectGuestDTO])
def list_project_guests(
    project_id: int,
    user: dict = Depends(get_current_user),
):
    """List guests for a project. Owner or project guests may view."""
    user_id = get_user_id(user)
    email = _get_user_email(user)

    with Session(engine) as session:
        _ensure_guest_claim(session, user_id=user_id, email=email)

        project = session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        is_owner = (session.get(ProjectOwner, project_id) or None)
        owner_ok = bool(is_owner and is_owner.owner_id == user_id)

        guest_ok = False
        if not owner_ok:
            guest_ok = session.exec(
                select(ProjectGuest)
                .join(Guest, Guest.id == ProjectGuest.guest_id)
                .where(ProjectGuest.project_id == project_id)
                .where(
                    (Guest.external_user_id == user_id)
                    | ((Guest.email == email) if email is not None else false())
                )
            ).first() is not None

        if not owner_ok and not guest_ok:
            raise HTTPException(status_code=403, detail="Not authorized")

        rows = session.exec(
            select(ProjectGuest, Guest)
            .join(Guest, Guest.id == ProjectGuest.guest_id)
            .where(ProjectGuest.project_id == project_id)
            .order_by(ProjectGuest.created_at.desc())
        ).all()

        return [
            ProjectGuestDTO(
                project_id=project_id,
                guest=GuestDTO(id=g.id or 0, email=g.email),
                role=pg.role,
                created_at=pg.created_at,
            )
            for (pg, g) in rows
        ]


@app.delete("/projects/{project_id}/guests/{guest_id}")
def remove_project_guest(
    project_id: int,
    guest_id: int,
    user: dict = Depends(get_current_user),
):
    """Remove a guest from a project. Only the project owner can remove."""
    user_id = get_user_id(user)
    with Session(engine) as session:
        owner_row = session.get(ProjectOwner, project_id)
        if not owner_row or owner_row.owner_id != user_id:
            raise HTTPException(status_code=403, detail="Only the project owner can remove guests")

        link = session.exec(
            select(ProjectGuest)
            .where(ProjectGuest.project_id == project_id)
            .where(ProjectGuest.guest_id == guest_id)
        ).first()
        if not link:
            raise HTTPException(status_code=404, detail="Guest link not found")

        session.delete(link)
        session.commit()
        return {"status": "success"}


@app.post("/connections", response_model=ConnectionDTO)
def create_connection(
    payload: CreateConnectionRequest,
    user_id: str = Depends(get_user_id)
):
    """Create a connection for the authenticated user's project."""
    connection_type = payload.type.strip().lower()
    project_name = payload.project_name.strip()

    with Session(engine) as session:
        project = session.exec(
            select(Project)
            .join(ProjectOwner, ProjectOwner.project_id == Project.id)
            .where(Project.name == project_name)
            .where(ProjectOwner.owner_id == user_id)
        ).first()

        if not project:
            # If a project exists with the same name but is owned by someone else, block.
            existing_by_name = session.exec(select(Project).where(Project.name == project_name)).first()
            if existing_by_name:
                owner_row = session.get(ProjectOwner, existing_by_name.id)
                if owner_row and owner_row.owner_id != user_id:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="A project with this name already exists for a different user.",
                    )
                project = existing_by_name
            else:
                project = Project(name=project_name, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
                session.add(project)
                session.commit()
                session.refresh(project)

            # Ensure ownership row exists for this user.
            if not session.get(ProjectOwner, project.id):
                session.add(ProjectOwner(project_id=project.id, owner_id=user_id))
                session.commit()

        if connection_type in {"repository", "repo", "github"}:
            if not payload.repo_url:
                raise HTTPException(status_code=400, detail="repo_url is required for repository connections")

            repo_name = _parse_repo_name(payload.repo_url)
            
            # Check if THIS user already has this repo
            repo = session.exec(
                select(Repository)
                .where(Repository.owner_id == user_id)
                .where(Repository.url == payload.repo_url)
            ).first()

            if not repo:
                # Not found for this user, create new (even if URL exists for others)
                repo = Repository(url=payload.repo_url, name=repo_name, owner_id=user_id)
                session.add(repo)
                session.commit()
                session.refresh(repo)

            link = session.exec(
                select(ProjectRepository).where(
                    ProjectRepository.project_id == project.id,
                    ProjectRepository.repository_id == repo.id,
                )
            ).first()
            if not link:
                session.add(
                    ProjectRepository(project_id=project.id, repository_id=repo.id, is_primary=True)
                )

            # Optional: store GitHub token for the user (used by /analyze if configured).
            if payload.github_token:
                try:
                    encrypted = encrypt_secret(payload.github_token)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e
                session.add(
                    IntegrationCredential(
                        owner_id=user_id,
                        provider="github",
                        name="GitHub",
                        encrypted_secret=encrypted,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                )

            project.updated_at = datetime.utcnow()
            session.add(project)
            session.commit()

            return ConnectionDTO(
                id=repo.id or 0,
                type="Repository",
                name=repo.name,
                project=project.name,
                status="active",
                lastSync=_format_relative_time(repo.last_synced_at),
            )

        if connection_type in {"board", "linear"}:
            if not payload.linear_api_key:
                raise HTTPException(status_code=400, detail="linear_api_key is required for Linear connections")
            if not payload.linear_team_key:
                raise HTTPException(status_code=400, detail="linear_team_key is required for Linear connections")

            try:
                encrypted = encrypt_secret(payload.linear_api_key)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            credential = IntegrationCredential(
                owner_id=user_id,
                provider="linear",
                name="Linear",
                encrypted_secret=encrypted,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(credential)
            session.commit()
            session.refresh(credential)

            integration = session.exec(
                select(ProjectIntegration).where(
                    ProjectIntegration.project_id == project.id,
                    ProjectIntegration.provider == "linear",
                )
            ).first()
            if not integration:
                integration = ProjectIntegration(
                    project_id=project.id,
                    provider="linear",
                    credential_id=credential.id,
                    settings={"team_key": payload.linear_team_key},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            else:
                integration.credential_id = credential.id
                integration.settings = {"team_key": payload.linear_team_key}
                integration.updated_at = datetime.utcnow()

            session.add(integration)
            project.updated_at = datetime.utcnow()
            session.add(project)
            session.commit()
            session.refresh(integration)

            return ConnectionDTO(
                id=integration.id or 0,
                type="Board",
                name=f"Linear ({payload.linear_team_key})",
                project=project.name,
                status="active",
                lastSync=_format_relative_time(integration.updated_at),
            )

        raise HTTPException(status_code=400, detail=f"Unsupported connection type: {payload.type}")


# --- Endpoint 4.5: SYNC (Background Ingestion) ---
@app.post("/sync", response_model=SyncRunDTO)
async def start_sync(
    payload: SyncRequest,
    user_id: str = Depends(get_user_id),
):
    """
    Starts a background sync run to ingest data from external providers into the DB.

    This endpoint is intentionally separate from `/analyze`:
    - `/sync` = ingest + persist (fast, no LLM)
    - `/analyze` = ingest + LLM + report (slow, expensive)
    """

    providers = [p.strip().lower() for p in (payload.providers or []) if p and p.strip()]
    allowed = {"github", "linear"}
    if not providers:
        providers = ["github", "linear"]
    if any(p not in allowed for p in providers):
        raise HTTPException(status_code=400, detail=f"Unsupported providers: {providers}")

    project_name = payload.project_name.strip()

    repo_name = None
    if payload.repo_name:
        try:
            repo_name = _parse_repo_name(payload.repo_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    if "github" in providers and not repo_name:
        raise HTTPException(
            status_code=400,
            detail="repo_name is required for GitHub sync (expected 'owner/repo' or a GitHub URL).",
        )

    # Note: enqueue_sync_run may raise ValueError on ownership/name conflicts.
    # We translate that into a user-facing 409.
    try:
        run = enqueue_sync_run(
            owner_id=user_id,
            project_name=project_name,
            repo_name=repo_name,
            providers=providers,
            full_history=payload.full_history,
            max_commits=payload.max_commits,
            max_prs=payload.max_prs,
            max_tickets=payload.max_tickets,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e

    return SyncRunDTO(
        id=run.id or 0,
        status=run.status,
        provider=run.provider,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        progress_current=run.progress_current,
        progress_total=run.progress_total,
        message=run.message,
        details=run.details or {},
    )


@app.get("/sync/runs", response_model=List[SyncRunDTO])
def list_sync_runs(
    limit: int = Query(default=50, ge=1, le=200),
    user_id: str = Depends(get_user_id),
):
    with Session(engine) as session:
        runs = session.exec(
            select(SyncRun)
            .where(SyncRun.owner_id == user_id)
            .order_by(SyncRun.created_at.desc())
            .limit(limit)
        ).all()
        return [
            SyncRunDTO(
                id=r.id or 0,
                status=r.status,
                provider=r.provider,
                created_at=r.created_at,
                started_at=r.started_at,
                finished_at=r.finished_at,
                progress_current=r.progress_current,
                progress_total=r.progress_total,
                message=r.message,
                details=r.details or {},
            )
            for r in runs
        ]


@app.get("/sync/runs/{run_id}", response_model=SyncRunDTO)
def get_sync_run(
    run_id: int,
    user_id: str = Depends(get_user_id),
):
    with Session(engine) as session:
        run = session.exec(
            select(SyncRun)
            .where(SyncRun.id == run_id)
            .where(SyncRun.owner_id == user_id)
        ).first()
        if not run:
            raise HTTPException(status_code=404, detail="Sync run not found")

        return SyncRunDTO(
            id=run.id or 0,
            status=run.status,
            provider=run.provider,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            progress_current=run.progress_current,
            progress_total=run.progress_total,
            message=run.message,
            details=run.details or {},
        )


# --- Endpoint 5: CHAT THREADS & HISTORY ---
@app.get("/chat/threads", response_model=List[ChatThreadDTO])
def list_chat_threads(user_id: str = Depends(get_user_id)):
    """List chat threads owned by the authenticated user."""
    with Session(engine) as session:
        # Always scope to authenticated user
        statement = select(ChatThread).where(ChatThread.owner_id == user_id)
        threads = session.exec(statement.order_by(ChatThread.updated_at.desc())).all()
        return [
            ChatThreadDTO(
                thread_id=t.external_thread_id,
                title=t.title or "Conversation",
                updated_at=t.updated_at,
            )
            for t in threads
        ]


@app.patch("/chat/threads/{thread_id}", response_model=ChatThreadDTO)
def update_chat_thread(
    thread_id: str,
    payload: dict = Body(...),
    user_id: str = Depends(get_user_id),
):
    """Update a chat thread's title (owned by the authenticated user)."""

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    title = payload.get("title")
    if title is None:
        raise HTTPException(status_code=400, detail="title is required")
    if not isinstance(title, str):
        raise HTTPException(status_code=400, detail="title must be a string")

    title = title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title must be non-empty")
    if len(title) > 255:
        raise HTTPException(status_code=400, detail="title must be <= 255 characters")

    with Session(engine) as session:
        thread = session.exec(
            select(ChatThread)
            .where(ChatThread.external_thread_id == thread_id)
            .where(ChatThread.owner_id == user_id)
        ).first()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        thread.title = title
        thread.updated_at = datetime.utcnow()
        session.add(thread)
        session.commit()
        session.refresh(thread)

        return ChatThreadDTO(
            thread_id=thread.external_thread_id,
            title=thread.title or "Conversation",
            updated_at=thread.updated_at,
        )


@app.get("/chat/threads/{thread_id}/messages", response_model=List[ChatMessageDTO])
def list_chat_messages(
    thread_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    user_id: str = Depends(get_user_id)
):
    """List messages from a thread owned by the authenticated user."""
    with Session(engine) as session:
        thread = session.exec(
            select(ChatThread)
            .where(ChatThread.external_thread_id == thread_id)
            .where(ChatThread.owner_id == user_id)
        ).first()
        if not thread:
            # Idempotent behavior: a new local thread may not exist in DB yet.
            return []

        messages = session.exec(
            select(ChatMessage)
            .where(ChatMessage.chat_thread_id == thread.id)
            .order_by(ChatMessage.id.asc())
            .limit(limit)
        ).all()

        return [
            ChatMessageDTO(
                id=m.id or 0,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
                metadata=m.message_metadata or {},
            )
            for m in messages
        ]


@app.delete("/chat/threads/{thread_id}")
def delete_chat_thread(
    thread_id: str,
    user_id: str = Depends(get_user_id)
):
    """Delete a chat thread and all its messages."""
    with Session(engine) as session:
        thread = session.exec(
            select(ChatThread)
            .where(ChatThread.external_thread_id == thread_id)
            .where(ChatThread.owner_id == user_id)
        ).first()
        
        if not thread:
            # Idempotent delete.
            return {"status": "success", "message": "Thread deleted"}

        # Delete messages first (manual cascade)
        messages = session.exec(
            select(ChatMessage).where(ChatMessage.chat_thread_id == thread.id)
        ).all()
        for m in messages:
            session.delete(m)

        # Delete thread
        session.delete(thread)
        session.commit()

        return {"status": "success", "message": "Thread deleted"}


# --- Endpoint 6: REPORTS ---
@app.get("/reports", response_model=List[ReportDTO])
def list_reports(
    project_name: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    user_id: str = Depends(get_user_id)
):
    """List analysis reports for repositories owned by the authenticated user."""
    with Session(engine) as session:
        # Scope to user's repositories only
        report_rows = session.exec(
            select(AnalysisReport)
            .join(Repository, AnalysisReport.repository_id == Repository.id)
            .where(Repository.owner_id == user_id)
            .order_by(AnalysisReport.created_at.desc())
            .limit(limit)
        ).all()

        reports: List[ReportDTO] = []
        for r in report_rows:
            repo = session.get(Repository, r.repository_id)
            repo_name = repo.name if repo else "unknown"

            # Resolve project association
            link = session.exec(
                select(ProjectRepository).where(ProjectRepository.repository_id == r.repository_id)
            ).first()
            
            project_label = "unknown"
            if link:
                project = session.get(Project, link.project_id)
                if project:
                    project_label = project.name

            # Filter by project_name if provided
            if project_name and project_label != project_name:
                continue

            created = r.created_at
            week = created.date().isoformat()

            status = "watch"
            if r.security_alerts:
                status = "at-risk"
            elif r.quality_score >= 7:
                status = "healthy"
            elif r.quality_score <= 3:
                status = "at-risk"

            # summary = (r.feedback_summary or "").strip().replace("\n", " ")
            # if len(summary) > 120:
            #     summary = summary[:120] + "..."
            
            # Use full summary without truncation
            summary = (r.feedback_summary or "").strip()

            reports.append(
                ReportDTO(
                    id=r.id or 0,
                    week=f"Week of {week}",
                    project=project_label,
                    repository=repo_name,
                    status=status,
                    summary=summary,
                    created_at=created,
                )
            )

        return reports


@app.delete("/reports/{report_id}")
def delete_report(
    report_id: int,
    user_id: str = Depends(get_user_id),
):
    with Session(engine) as session:
        report = session.exec(
            select(AnalysisReport)
            .join(Repository, AnalysisReport.repository_id == Repository.id)
            .where(AnalysisReport.id == report_id)
            .where(Repository.owner_id == user_id)
        ).first()
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        session.delete(report)
        session.commit()
        return {"status": "success", "message": "Report deleted"}


@app.delete("/connections/{connection_id}")
def delete_connection(
    connection_id: int,
    type: str = Query(..., description="Type of connection: 'Repository' or 'Board'"),
    user_id: str = Depends(get_user_id),
):
    with Session(engine) as session:
        if type.lower() == "repository":
            repo = session.exec(
                select(Repository)
                .where(Repository.id == connection_id)
                .where(Repository.owner_id == user_id)
            ).first()
            if not repo:
                raise HTTPException(status_code=404, detail="Connection not found")

            # Find any project links to this repo and remove them
            links = session.exec(select(ProjectRepository).where(ProjectRepository.repository_id == connection_id)).all()
            if not links:
                return {"status": "success", "message": "Repository connection removed"}
            for link in links:
                session.delete(link)
            session.commit()
            return {"status": "success", "message": "Repository connection removed"}
            
        elif type.lower() == "board":
            # Handle synthetic IDs or missing integration records
            project_id = None
            integration = session.get(ProjectIntegration, connection_id)
            
            if integration:
                credential_owner_id = None
                if integration.credential_id:
                    credential = session.get(IntegrationCredential, integration.credential_id)
                    if credential:
                        credential_owner_id = credential.owner_id
                if credential_owner_id and credential_owner_id != user_id:
                    raise HTTPException(status_code=404, detail="Connection not found")
                project_id = integration.project_id
                session.delete(integration)
            elif connection_id >= 1000000:
                # Fallback for synthetic IDs generated in list_connections
                project_id = connection_id - 1000000
                # Try to find integration by project_id just in case
                integration = session.exec(select(ProjectIntegration).where(
                    ProjectIntegration.project_id == project_id,
                    ProjectIntegration.provider == "linear"
                )).first()
                if integration:
                    credential_owner_id = None
                    if integration.credential_id:
                        credential = session.get(IntegrationCredential, integration.credential_id)
                        if credential:
                            credential_owner_id = credential.owner_id
                    if credential_owner_id and credential_owner_id != user_id:
                        raise HTTPException(status_code=404, detail="Connection not found")
                    session.delete(integration)
            
            if not project_id:
                 raise HTTPException(status_code=404, detail="Connection not found")

            # Also delete tickets to ensure it disappears from the list (which shows ghost connections if tickets exist)
            tickets = session.exec(select(Ticket).where(
                Ticket.project_id == project_id,
                Ticket.source_platform == "linear"
            )).all()
            ticket_ids = [t.id for t in tickets if t.id is not None]
            
            for t in tickets:
                session.delete(t)

            if ticket_ids:
                events = session.exec(
                    select(TicketEvent).where(TicketEvent.ticket_id.in_(ticket_ids))
                ).all()
                for e in events:
                    session.delete(e)

            session.commit()
            return {"status": "success", "message": "Board connection removed"}
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown connection type: {type}")
