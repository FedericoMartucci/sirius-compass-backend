# app/api/server.py
import asyncio
import os
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Request, status
from fastapi.responses import StreamingResponse
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Depends, Query
# from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from sqlmodel import Session, select

# Import Schemas from your existing file
from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatMessageDTO,
    ChatRequest,
    ChatThreadDTO,
    ConnectionDTO,
    CreateConnectionRequest,
    CreateProjectRequest,
    ProjectDTO,
    ReportDTO,
    SyncRequest,
    SyncRunDTO,
)

from app.core.database.session import create_db_and_tables, engine
from app.core.agents.analyst_graph import build_analyst_graph
from app.core.agents.chat_graph import build_chat_graph
from app.core.logger import get_logger
from app.core.security.crypto import encrypt_secret
from app.core.security.auth import get_user_id
from app.core.streaming import TokenStreamHandler, sse_data
from app.core.database.models import (
    AnalysisReport,
    ChatMessage,
    ChatThread,
    DataCoverage,
    IntegrationCredential,
    Project,
    ProjectIntegration,
    ProjectRepository,
    Repository,
    SyncRun,
    Ticket,
    TicketEvent,
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
def list_projects(user_id: str = Depends(get_user_id)):
    """List projects owned by the authenticated user."""
    with Session(engine) as session:
        # Scope to user's projects via ProjectRepository ownership
        projects = session.exec(
            select(Project)
            .join(ProjectRepository, Project.id == ProjectRepository.project_id)
            .join(Repository, ProjectRepository.repository_id == Repository.id)
            .where(Repository.owner_id == user_id)
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
        # Check for existing project owned by this user
        existing = session.exec(
            select(Project)
            .join(ProjectRepository, Project.id == ProjectRepository.project_id)
            .join(Repository, ProjectRepository.repository_id == Repository.id)
            .where(Project.name == name)
            .where(Repository.owner_id == user_id)
        ).first()
        if existing:
            return ProjectDTO(id=str(existing.id), name=existing.name)

        project = Project(name=name, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        session.add(project)
        session.commit()
        session.refresh(project)
        return ProjectDTO(id=str(project.id), name=project.name)


# --- Endpoint 4: CONNECTIONS (Integrations) ---
@app.get("/connections", response_model=List[ConnectionDTO])
def list_connections(
    project_name: Optional[str] = Query(default=None),
    user_id: str = Depends(get_user_id)
):
    """List connections scoped to authenticated user's projects."""
    with Session(engine) as session:
        # Fetch LATEST sync run for every scope (regardless of status)
        # We need to do this efficiently. For now, we'll fetch recent runs and filter in memory since we scope by user.
        # A better SQL approach would be distinct on (repo_id) order by created_at desc, but SQLModel is tricky with that.
        all_recent_runs = session.exec(
            select(SyncRun)
            .where(SyncRun.owner_id == user_id)
            .order_by(SyncRun.created_at.desc())
            .limit(500)  # reasonable window
        ).all()
        
        latest_repo_run: dict[int, SyncRun] = {}
        latest_project_run: dict[int, SyncRun] = {}
        
        for r in all_recent_runs:
            if r.repository_id and r.repository_id not in latest_repo_run:
                latest_repo_run[r.repository_id] = r
            if r.project_id and r.project_id not in latest_project_run and r.provider in {"linear", "all"}:
                latest_project_run[r.project_id] = r

        def get_status_from_run(run: Optional[SyncRun]) -> tuple[str, Optional[str]]:
            if not run:
                return "active", None
            if run.status in {"queued", "running"}:
                return "syncing", None
            if run.status == "failed":
                return "error", run.message
            return "active", None

        projects: List[Project] = []
        # Only fetch projects that have repositories owned by this user
        base_query = (
            select(Project)
            .join(ProjectRepository, Project.id == ProjectRepository.project_id)
            .join(Repository, ProjectRepository.repository_id == Repository.id)
            .where(Repository.owner_id == user_id)
        )
        
        if project_name:
            project = session.exec(base_query.where(Project.name == project_name)).first()
            if project:
                projects = [project]
        else:
            projects = session.exec(base_query).all()

        connections: List[ConnectionDTO] = []

        # Repository connections
        for project in projects:
            links = session.exec(
                select(ProjectRepository).where(ProjectRepository.project_id == project.id)
            ).all()
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


@app.post("/connections", response_model=ConnectionDTO)
def create_connection(
    payload: CreateConnectionRequest,
    user_id: str = Depends(get_user_id)
):
    """Create a connection for the authenticated user's project."""
    connection_type = payload.type.strip().lower()
    project_name = payload.project_name.strip()

    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name == project_name)).first()
        if not project:
            project = Project(name=project_name, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
            session.add(project)
            session.commit()
            session.refresh(project)

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
