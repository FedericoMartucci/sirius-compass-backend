import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, select

from app.adapters.linear.adapter import LinearAdapter, LinearRateLimitError
from app.core.database.models import (
    IntegrationCredential,
    LinearRateLimitBucket,
    ProjectIntegration,
    ProjectRepository,
    Repository,
    SyncJob,
    Ticket,
    TicketEvent,
)
from app.core.database.session import engine
from app.core.logger import get_logger
from app.core.security.crypto import decrypt_secret
from app.services.sync import SyncService

logger = get_logger(__name__)

LINEAR_BUCKET_CAPACITY = 5
LINEAR_BUCKET_WINDOW_SECONDS = 300


def new_ticket_id(prefix: str = "syncjob") -> str:
    return f"{prefix}_{uuid.uuid4()}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).isoformat()
    return dt.astimezone(timezone.utc).isoformat()


def _append_snapshot(job: SyncJob, *, message: str, data: Optional[Dict[str, Any]] = None) -> None:
    snapshot = {
        "at": _dt_to_iso(utcnow()),
        "message": message,
    }
    if data:
        snapshot["data"] = data
    job.snapshots = list(job.snapshots or [])
    job.snapshots.append(snapshot)


def _linear_bucket_acquire(session: Session, credential_id: int) -> Tuple[bool, Optional[datetime]]:
    """Returns (allowed, next_allowed_at)."""
    now = utcnow()
    bucket = session.get(LinearRateLimitBucket, credential_id)
    if not bucket:
        bucket = LinearRateLimitBucket(credential_id=credential_id, window_started_at=now, tokens_used=0)
        session.add(bucket)
        session.commit()
        session.refresh(bucket)

    window_started = bucket.window_started_at
    if window_started is None:
        bucket.window_started_at = now
        bucket.tokens_used = 0
        session.add(bucket)
        session.commit()
        return True, None

    if window_started.tzinfo is None:
        window_started = window_started.replace(tzinfo=timezone.utc)

    window_end = window_started + timedelta(seconds=LINEAR_BUCKET_WINDOW_SECONDS)
    if now >= window_end:
        bucket.window_started_at = now
        bucket.tokens_used = 0
        session.add(bucket)
        session.commit()
        return True, None

    if bucket.tokens_used >= LINEAR_BUCKET_CAPACITY:
        return False, window_end

    bucket.tokens_used += 1
    session.add(bucket)
    session.commit()
    return True, None


def create_job(*, owner_id: str, connection_type: str, connection_id: int) -> SyncJob:
    ticket = new_ticket_id("syncjob")
    job = SyncJob(
        id=ticket,
        owner_id=owner_id,
        connection_type=connection_type,
        connection_id=connection_id,
        state="queued",
        progress={"started": _dt_to_iso(utcnow())},
        snapshots=[],
        next_run_at=None,
        error=None,
    )
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
    return job


def schedule_job(ticket: str) -> None:
    """Fire-and-forget scheduling from an async FastAPI request context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop (e.g., tests) – run inline.
        run_job(ticket)
        return

    loop.create_task(asyncio.to_thread(run_job, ticket))


def get_job(ticket: str, owner_id: str) -> Optional[SyncJob]:
    with Session(engine) as session:
        job = session.get(SyncJob, ticket)
        if not job or job.owner_id != owner_id:
            return None
        return job


def maybe_resume_job(ticket: str, owner_id: str) -> None:
    with Session(engine) as session:
        job = session.get(SyncJob, ticket)
        if not job or job.owner_id != owner_id:
            return

        if job.state != "waiting_rate_limit":
            return

        if job.next_run_at and job.next_run_at.tzinfo is None:
            job.next_run_at = job.next_run_at.replace(tzinfo=timezone.utc)

        if job.next_run_at and utcnow() < job.next_run_at:
            return

        job.state = "queued"
        job.next_run_at = None
        job.updated_at = datetime.utcnow()
        _append_snapshot(job, message="Rate limit window ended; re-queued")
        session.add(job)
        session.commit()

    schedule_job(ticket)


def run_job(ticket: str) -> None:
    """Executes a SyncJob; safe to call multiple times."""
    with Session(engine) as session:
        job = session.get(SyncJob, ticket)
        if not job:
            return

        if job.state in {"running", "completed", "failed"}:
            return

        if job.state == "waiting_rate_limit":
            if job.next_run_at and job.next_run_at.tzinfo is None:
                job.next_run_at = job.next_run_at.replace(tzinfo=timezone.utc)
            if job.next_run_at and utcnow() < job.next_run_at:
                return

        job.state = "running"
        job.error = None
        job.updated_at = datetime.utcnow()
        _append_snapshot(job, message="Job started")
        session.add(job)
        session.commit()

    try:
        if job.connection_type == "repository":
            _run_repository_sync(ticket)
        elif job.connection_type == "board":
            _run_board_sync(ticket)
        else:
            raise ValueError(f"Unsupported connection_type: {job.connection_type}")

        with Session(engine) as session:
            job = session.get(SyncJob, ticket)
            if not job:
                return
            job.state = "completed"
            job.updated_at = datetime.utcnow()
            job.progress = {**(job.progress or {}), "completed": _dt_to_iso(utcnow())}
            _append_snapshot(job, message="Job completed")
            session.add(job)
            session.commit()

    except _JobWaitRateLimit as e:
        with Session(engine) as session:
            job = session.get(SyncJob, ticket)
            if not job:
                return
            job.state = "waiting_rate_limit"
            job.next_run_at = e.next_run_at
            job.updated_at = datetime.utcnow()
            job.progress = {**(job.progress or {}), "waiting": _dt_to_iso(utcnow())}
            _append_snapshot(
                job,
                message="Waiting for rate limit window",
                data={"next_run_at": _dt_to_iso(e.next_run_at)},
            )
            session.add(job)
            session.commit()

    except Exception as e:
        logger.exception("Sync job failed")
        with Session(engine) as session:
            job = session.get(SyncJob, ticket)
            if not job:
                return
            job.state = "failed"
            job.error = str(e)
            job.updated_at = datetime.utcnow()
            _append_snapshot(job, message="Job failed", data={"error": str(e)})
            session.add(job)
            session.commit()


class _JobWaitRateLimit(Exception):
    def __init__(self, next_run_at: datetime):
        self.next_run_at = next_run_at


def _resolve_repository_for_user(session: Session, *, repo_id: int, owner_id: str) -> Repository:
    repo = session.exec(
        select(Repository)
        .where(Repository.id == repo_id)
        .where(Repository.owner_id == owner_id)
    ).first()
    if not repo:
        raise ValueError("Repository not found or access denied")
    return repo


def _validate_project_access(session: Session, *, project_id: int, owner_id: str) -> None:
    has_access = session.exec(
        select(ProjectRepository.project_id)
        .join(Repository, ProjectRepository.repository_id == Repository.id)
        .where(ProjectRepository.project_id == project_id)
        .where(Repository.owner_id == owner_id)
        .limit(1)
    ).first() is not None
    if not has_access:
        raise ValueError("Project not found or access denied")


def _resolve_linear_integration_for_user(
    session: Session,
    *,
    connection_id: int,
    owner_id: str,
) -> ProjectIntegration:
    project_id: Optional[int] = None

    integration = session.get(ProjectIntegration, connection_id)
    if integration and integration.provider != "linear":
        integration = None

    if integration:
        project_id = integration.project_id
    elif connection_id >= 1_000_000:
        project_id = connection_id - 1_000_000
        integration = session.exec(
            select(ProjectIntegration)
            .where(ProjectIntegration.project_id == project_id)
            .where(ProjectIntegration.provider == "linear")
            .order_by(ProjectIntegration.updated_at.desc())
        ).first()

    if not project_id:
        raise ValueError("Board connection not found")

    _validate_project_access(session, project_id=project_id, owner_id=owner_id)

    if not integration:
        raise ValueError("Linear integration not configured for this project")

    return integration


def _run_repository_sync(ticket: str) -> None:
    with Session(engine) as session:
        job = session.get(SyncJob, ticket)
        if not job:
            return

        repo = _resolve_repository_for_user(session, repo_id=job.connection_id, owner_id=job.owner_id)

        service = SyncService(session)
        # NOTE: SyncService isn't multi-tenant-safe by url; enforce by using the resolved repo.url.
        service.ensure_repository_sync(repo.url, days_lookback=7)

        job.progress = {**(job.progress or {}), "repository": repo.name, "lastSyncAt": _dt_to_iso(repo.last_synced_at)}
        _append_snapshot(job, message="Repository sync finished", data={"repo": repo.name})
        session.add(job)
        session.commit()


def _run_board_sync(ticket: str) -> None:
    with Session(engine) as session:
        job = session.get(SyncJob, ticket)
        if not job:
            return

        integration = _resolve_linear_integration_for_user(
            session,
            connection_id=job.connection_id,
            owner_id=job.owner_id,
        )

        team_key = (integration.settings or {}).get("team_key")
        if not team_key:
            raise ValueError("Linear team_key missing in ProjectIntegration.settings")

        if not integration.credential_id:
            raise ValueError("Linear credential not configured")

        credential = session.get(IntegrationCredential, integration.credential_id)
        if not credential or credential.owner_id != job.owner_id or credential.provider != "linear":
            raise ValueError("Linear credential not found or access denied")

        allowed, next_allowed_at = _linear_bucket_acquire(session, credential.id)
        if not allowed:
            raise _JobWaitRateLimit(next_run_at=next_allowed_at or (utcnow() + timedelta(seconds=LINEAR_BUCKET_WINDOW_SECONDS)))

        api_key = decrypt_secret(credential.encrypted_secret)
        adapter = LinearAdapter(api_key=api_key)

        # Incremental: use latest known updated_at for this project.
        latest = session.exec(
            select(Ticket)
            .where(Ticket.project_id == integration.project_id)
            .where(Ticket.source_platform == "linear")
            .order_by(Ticket.updated_at.desc().nullslast(), Ticket.created_at.desc())
            .limit(1)
        ).first()

        updated_since: Optional[datetime] = None
        if latest:
            updated_since = latest.updated_at or latest.created_at
            if updated_since and updated_since.tzinfo is None:
                updated_since = updated_since.replace(tzinfo=timezone.utc)

        _append_snapshot(
            job,
            message="Fetching Linear issues",
            data={"team_key": team_key, "updated_since": _dt_to_iso(updated_since)},
        )
        session.add(job)
        session.commit()

        try:
            issues, meta = adapter.fetch_recent_issues_with_meta(limit=200, team_key=team_key, updated_since=updated_since)
        except LinearRateLimitError as e:
            raise _JobWaitRateLimit(next_run_at=utcnow() + timedelta(seconds=e.retry_after_seconds))

        new_tickets = 0
        updated_tickets = 0
        new_events = 0

        for item in issues:
            # Timestamps
            created_at = datetime.fromisoformat(item.createdAt.replace("Z", "+00:00"))
            updated_at = None
            if item.updatedAt:
                try:
                    updated_at = datetime.fromisoformat(item.updatedAt.replace("Z", "+00:00"))
                except Exception:
                    updated_at = None

            completed_at = None
            if item.completedAt:
                try:
                    completed_at = datetime.fromisoformat(item.completedAt.replace("Z", "+00:00"))
                except Exception:
                    completed_at = None

            incoming_points = int(getattr(item, "estimate", 0) or 0)
            incoming_status = getattr(item, "state", None)
            incoming_key = getattr(item, "identifier", None)

            ticket_db = session.exec(select(Ticket).where(Ticket.source_id == item.id)).first()
            if not ticket_db:
                ticket_db = Ticket(
                    project_id=integration.project_id,
                    source_platform="linear",
                    source_id=item.id,
                    key=incoming_key,
                    title=item.title or "Untitled",
                    description=item.description,
                    story_points=incoming_points,
                    status_label=incoming_status,
                    created_at=created_at,
                    updated_at=updated_at or utcnow(),
                    completed_at=completed_at,
                    url=item.url,
                    raw_payload={"meta": meta},
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
                new_tickets += 1
                new_events += 1
                continue

            # Ensure correct project binding.
            if ticket_db.project_id != integration.project_id:
                ticket_db.project_id = integration.project_id

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
                new_events += 1

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
                new_events += 1

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
                new_events += 1

            if item.title and item.title != ticket_db.title:
                session.add(
                    TicketEvent(
                        ticket_id=ticket_db.id,
                        event_type="title_changed",
                        from_value=ticket_db.title,
                        to_value=item.title,
                    )
                )
                ticket_db.title = item.title
                new_events += 1

            # Always update mutable fields.
            ticket_db.description = item.description
            ticket_db.updated_at = updated_at or utcnow()
            ticket_db.completed_at = completed_at
            ticket_db.url = item.url
            ticket_db.raw_payload = {
                **(ticket_db.raw_payload or {}),
                "external_key": incoming_key,
                "status_label": incoming_status,
                "story_points": incoming_points,
            }

            updated_tickets += 1

        session.commit()

        job.progress = {
            **(job.progress or {}),
            "board": {
                "project_id": integration.project_id,
                "team_key": team_key,
                "updated_since": _dt_to_iso(updated_since),
                "fetched": len(issues),
                "newTickets": new_tickets,
                "updatedTickets": updated_tickets,
                "newEvents": new_events,
            },
        }
        _append_snapshot(
            job,
            message="Linear sync finished",
            data={"fetched": len(issues), "new": new_tickets, "updated": updated_tickets, "events": new_events},
        )
        session.add(job)
        session.commit()
