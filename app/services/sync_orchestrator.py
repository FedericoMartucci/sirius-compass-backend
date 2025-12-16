from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from sqlmodel import Session, select

from app.adapters.github.adapter import GitHubAdapter
from app.adapters.linear.adapter import LinearAdapter, LinearIssue
from app.core.database.models import (
    Activity,
    DataCoverage,
    IntegrationCredential,
    Project,
    ProjectOwner,
    ProjectIntegration,
    ProjectRepository,
    Repository,
    SyncRun,
    Ticket,
    TicketEvent,
)
from app.core.database.session import engine
from app.core.logger import get_logger
from app.core.models.domain import ActivityType
from app.core.security.crypto import decrypt_secret

logger = get_logger(__name__)


def _chunked(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _ticket_fingerprint(issue: LinearIssue) -> str:
    payload = "|".join(
        [
            issue.identifier or "",
            issue.title or "",
            (issue.description or ""),
            str(issue.estimate or 0),
            issue.state or "",
            issue.state_type or "",
            issue.assignee or "",
            issue.updatedAt or "",
            issue.completedAt or "",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _set_run_status(
    session: Session,
    *,
    run: SyncRun,
    status: str,
    message: Optional[str] = None,
    progress_current: Optional[int] = None,
    progress_total: Optional[int] = None,
) -> None:
    run.status = status
    run.updated_at = datetime.utcnow()
    if status == "running" and run.started_at is None:
        run.started_at = datetime.utcnow()
    if status in {"completed", "failed"}:
        run.finished_at = datetime.utcnow()
    if message is not None:
        run.message = message
    if progress_current is not None:
        run.progress_current = progress_current
    if progress_total is not None:
        run.progress_total = progress_total
    session.add(run)


def _upsert_coverage(
    session: Session,
    *,
    owner_id: str,
    scope_type: str,
    scope_id: int,
    provider: str,
    earliest_at: Optional[datetime],
    latest_at: Optional[datetime],
    is_complete: Optional[bool],
    last_run_id: int,
    notes: Optional[str] = None,
) -> None:
    row = session.exec(
        select(DataCoverage)
        .where(DataCoverage.owner_id == owner_id)
        .where(DataCoverage.scope_type == scope_type)
        .where(DataCoverage.scope_id == scope_id)
        .where(DataCoverage.provider == provider)
    ).first()

    if not row:
        row = DataCoverage(
            owner_id=owner_id,
            scope_type=scope_type,
            scope_id=scope_id,
            provider=provider,
            earliest_at=earliest_at,
            latest_at=latest_at,
            is_complete=bool(is_complete),
            updated_at=datetime.utcnow(),
            last_run_id=last_run_id,
            notes=notes,
        )
        session.add(row)
        return

    if earliest_at is not None:
        earliest_at_utc = _to_utc(earliest_at)
        existing_earliest = _to_utc(row.earliest_at) if row.earliest_at else None
        if existing_earliest is None or (earliest_at_utc and earliest_at_utc < existing_earliest):
            row.earliest_at = earliest_at_utc
    if latest_at is not None:
        latest_at_utc = _to_utc(latest_at)
        existing_latest = _to_utc(row.latest_at) if row.latest_at else None
        if existing_latest is None or (latest_at_utc and latest_at_utc > existing_latest):
            row.latest_at = latest_at_utc

    if is_complete is True:
        row.is_complete = True
    row.updated_at = datetime.utcnow()
    row.last_run_id = last_run_id
    if notes:
        row.notes = notes
    session.add(row)


class SyncOrchestrator:
    """
    Background sync runner used by API endpoints and the chat agent.

    Notes:
    - This is intentionally synchronous: it runs inside `asyncio.to_thread(...)`.
    - It commits progress frequently so the UI can poll progress.
    """

    def __init__(self, *, owner_id: str, run_id: int) -> None:
        self.owner_id = owner_id
        self.run_id = run_id

    def run(
        self,
        *,
        project_name: str,
        repo_name: Optional[str],
        providers: list[str],
        full_history: bool,
        max_commits: Optional[int],
        max_prs: Optional[int],
        max_tickets: Optional[int],
    ) -> None:
        with Session(engine) as session:
            run = session.get(SyncRun, self.run_id)
            if not run or run.owner_id != self.owner_id:
                return

            _set_run_status(session, run=run, status="running", message="Starting sync...")
            session.commit()

        try:
            project_id = self._ensure_project(project_name)
            repo_id: Optional[int] = None
            if repo_name:
                repo_id = self._ensure_repository(repo_name, project_id)

            if "github" in providers:
                if repo_id is None:
                    raise ValueError("repo_name is required for GitHub sync.")
                self._sync_github_repository(
                    repository_id=repo_id,
                    full_history=full_history,
                    max_commits=max_commits,
                    max_prs=max_prs,
                )

            if "linear" in providers:
                self._sync_linear_project(
                    project_id=project_id,
                    full_history=full_history,
                    max_tickets=max_tickets,
                )

            with Session(engine) as session:
                run = session.get(SyncRun, self.run_id)
                if run:
                    _set_run_status(session, run=run, status="completed", message="Sync completed.")
                    session.commit()
        except Exception as e:
            logger.exception("Sync run failed")
            with Session(engine) as session:
                run = session.get(SyncRun, self.run_id)
                if run:
                    _set_run_status(session, run=run, status="failed", message=str(e))
                    session.commit()

    def _ensure_project(self, project_name: str) -> int:
        with Session(engine) as session:
            project = session.exec(
                select(Project)
                .join(ProjectOwner, ProjectOwner.project_id == Project.id)
                .where(Project.name == project_name)
                .where(ProjectOwner.owner_id == self.owner_id)
            ).first()

            if not project:
                existing_by_name = session.exec(select(Project).where(Project.name == project_name)).first()
                if existing_by_name:
                    owner_row = session.get(ProjectOwner, existing_by_name.id)
                    if owner_row and owner_row.owner_id != self.owner_id:
                        raise ValueError("Project name already exists for a different user.")
                    project = existing_by_name
                else:
                    project = Project(name=project_name, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
                    session.add(project)
                    session.commit()
                    session.refresh(project)

                if not session.get(ProjectOwner, project.id):
                    session.add(ProjectOwner(project_id=project.id, owner_id=self.owner_id))
                    session.commit()
            return project.id or 0

    def _ensure_repository(self, repo_name: str, project_id: int) -> int:
        with Session(engine) as session:
            repo = session.exec(
                select(Repository)
                .where(Repository.owner_id == self.owner_id)
                .where(Repository.name == repo_name)
            ).first()
            if not repo:
                repo = Repository(
                    url=f"https://github.com/{repo_name}",
                    name=repo_name,
                    owner_id=self.owner_id,
                )
                session.add(repo)
                session.commit()
                session.refresh(repo)

            link = session.exec(
                select(ProjectRepository).where(
                    ProjectRepository.project_id == project_id,
                    ProjectRepository.repository_id == repo.id,
                )
            ).first()
            if not link:
                session.add(
                    ProjectRepository(project_id=project_id, repository_id=repo.id, is_primary=True)
                )
                session.commit()
            return repo.id or 0

    def _sync_github_repository(
        self,
        *,
        repository_id: int,
        full_history: bool,
        max_commits: Optional[int],
        max_prs: Optional[int],
    ) -> None:
        with Session(engine) as session:
            repo = session.exec(
                select(Repository)
                .where(Repository.id == repository_id)
                .where(Repository.owner_id == self.owner_id)
            ).first()
            if not repo:
                raise ValueError("Repository not found.")

            github_cred = session.exec(
                select(IntegrationCredential)
                .where(
                    IntegrationCredential.owner_id == self.owner_id,
                    IntegrationCredential.provider == "github",
                )
                .order_by(IntegrationCredential.updated_at.desc())
            ).first()
            token = decrypt_secret(github_cred.encrypted_secret) if github_cred else None

        if not token:
            raise ValueError("Missing GitHub credentials. Please connect a GitHub token first.")

        now = datetime.now(timezone.utc)
        if full_history:
            days_candidates = [36500]
        elif repo.last_synced_at:
            last = _to_utc(repo.last_synced_at) or now
            days_candidates = [max((now - last).days + 1, 1)]
        else:
            # Initial sync: expand lookback until we find *something* (up to 3 years).
            days_candidates = [90, 180, 365, 730, 1095]

        with Session(engine) as session:
            run = session.get(SyncRun, self.run_id)
            if run:
                _set_run_status(
                    session,
                    run=run,
                    status="running",
                    message=f"Syncing GitHub ({days_candidates[0]}d lookback)...",
                    progress_current=0,
                    progress_total=(max_commits or 0) + (max_prs or 0) or None,
                )
                session.commit()

        gh = GitHubAdapter(token=token)
        result = None
        for attempt_idx, lookback_days in enumerate(days_candidates, start=1):
            with Session(engine) as session:
                run = session.get(SyncRun, self.run_id)
                if run and len(days_candidates) > 1:
                    _set_run_status(
                        session,
                        run=run,
                        status="running",
                        message=f"Syncing GitHub ({lookback_days}d lookback, attempt {attempt_idx}/{len(days_candidates)})...",
                        progress_current=0,
                        progress_total=(max_commits or 0) + (max_prs or 0) or None,
                    )
                    session.commit()

            result = gh.fetch_recent_activity_with_meta(
                repo.name,
                days=lookback_days,
                max_commits=max_commits,
                max_prs=max_prs,
                include_prs=True,
            )
            if result.commits_fetched or result.prs_fetched:
                break
        if result is None:
            raise ValueError("GitHub sync failed: no result produced.")

        with Session(engine) as session:
            repo_db = session.get(Repository, repository_id)
            if not repo_db:
                raise ValueError("Repository not found.")

            inserted = 0
            activities = result.activities

            source_ids = [a.source_id for a in activities]
            existing_source_ids: set[str] = set()
            for batch in _chunked(source_ids, 200):
                existing = session.exec(
                    select(Activity.source_id)
                    .where(Activity.repository_id == repo_db.id)
                    .where(Activity.source_id.in_(batch))
                ).all()
                existing_source_ids.update(existing)

            for act in activities:
                if act.source_id in existing_source_ids:
                    continue
                session.add(
                    Activity(
                        repository_id=repo_db.id,
                        source_platform="github",
                        source_id=act.source_id,
                        type=act.type.value if hasattr(act.type, "value") else str(act.type),
                        author=act.author,
                        timestamp=act.timestamp,
                        title=(act.title or (act.content.splitlines()[0].strip() if act.content else "Untitled"))[:255],
                        content=act.content,
                        files_changed_count=len(act.files_changed or []),
                        story_points=0,
                        status_label=None,
                    )
                )
                inserted += 1

            if result.latest_commit_at and (
                repo_db.last_synced_at is None or _to_utc(result.latest_commit_at) > _to_utc(repo_db.last_synced_at)
            ):
                repo_db.last_synced_at = _to_utc(result.latest_commit_at)
                session.add(repo_db)

            notes = None
            if result.commits_truncated:
                notes = "commits_truncated"
            elif not result.commits_fetched and not result.prs_fetched and not full_history and not repo_db.last_synced_at:
                notes = "no_activity_found_up_to_3y"

            _upsert_coverage(
                session,
                owner_id=self.owner_id,
                scope_type="repository",
                scope_id=repo_db.id,
                provider="github",
                earliest_at=_to_utc(result.earliest_commit_at),
                latest_at=_to_utc(result.latest_commit_at),
                is_complete=bool(full_history and not result.commits_truncated),
                last_run_id=self.run_id,
                notes=notes,
            )

            run = session.get(SyncRun, self.run_id)
            if run:
                _set_run_status(
                    session,
                    run=run,
                    status="running",
                    message=f"GitHub sync done. Inserted {inserted} new items.",
                    progress_current=len(activities),
                    progress_total=(max_commits or 0) + (max_prs or 0) or None,
                )
            session.commit()

    def _sync_linear_project(
        self,
        *,
        project_id: int,
        full_history: bool,
        max_tickets: Optional[int],
    ) -> None:
        with Session(engine) as session:
            project = session.get(Project, project_id)
            if not project:
                raise ValueError("Project not found.")

            integration = session.exec(
                select(ProjectIntegration)
                .where(ProjectIntegration.project_id == project_id)
                .where(ProjectIntegration.provider == "linear")
                .order_by(ProjectIntegration.updated_at.desc())
            ).first()

            if not integration or not integration.credential_id:
                raise ValueError("Missing Linear integration. Please connect Linear first.")

            credential = session.get(IntegrationCredential, integration.credential_id)
            if not credential:
                raise ValueError("Missing Linear credentials. Please connect Linear first.")
            if credential.owner_id and credential.owner_id != self.owner_id:
                raise ValueError("Linear credentials do not belong to the authenticated user.")

            api_key = decrypt_secret(credential.encrypted_secret)
            team_key = (integration.settings or {}).get("team_key")

            latest_ticket = session.exec(
                select(Ticket)
                .where(Ticket.project_id == project_id)
                .where(Ticket.source_platform == "linear")
                .order_by(Ticket.updated_at.desc())
            ).first()

        updated_since = None
        initial_no_tickets = latest_ticket is None
        if not initial_no_tickets and latest_ticket and latest_ticket.updated_at:
            updated_since = _to_utc(latest_ticket.updated_at)

        # If this is the first Linear sync for the project, do not apply a safety cap:
        # PM expectations are that an "initial sync" backfills the whole board.
        effective_max_tickets = None if initial_no_tickets else max_tickets
        should_full = full_history or initial_no_tickets

        with Session(engine) as session:
            run = session.get(SyncRun, self.run_id)
            if run:
                _set_run_status(
                    session,
                    run=run,
                    status="running",
                    message="Syncing Linear tickets...",
                    progress_current=0,
                    progress_total=effective_max_tickets,
                )
                session.commit()

        lin = LinearAdapter(api_key=api_key)

        total_fetched = 0
        total_written = 0
        cursor: Optional[str] = None
        truncated = False

        while True:
            page, next_cursor, has_next = lin.fetch_issues_page(
                limit=50,
                team_key=team_key,
                updated_since=updated_since,
                after=cursor,
            )

            if not page:
                break

            total_fetched += len(page)
            written = self._persist_linear_page(project_id=project_id, issues=page)
            total_written += written

            with Session(engine) as session:
                run = session.get(SyncRun, self.run_id)
                if run:
                    _set_run_status(
                        session,
                        run=run,
                        status="running",
                        message=f"Linear sync in progress... (fetched={total_fetched}, written={total_written})",
                        progress_current=total_fetched,
                        progress_total=effective_max_tickets,
                    )
                    session.commit()

            if effective_max_tickets is not None and total_fetched >= effective_max_tickets:
                truncated = True
                break

            if not has_next or not next_cursor:
                break
            cursor = next_cursor

        with Session(engine) as session:
            earliest_ticket = session.exec(
                select(Ticket.created_at)
                .where(Ticket.project_id == project_id)
                .where(Ticket.source_platform == "linear")
                .order_by(Ticket.created_at.asc())
                .limit(1)
            ).first()
            latest_ticket = session.exec(
                select(Ticket.updated_at)
                .where(Ticket.project_id == project_id)
                .where(Ticket.source_platform == "linear")
                .order_by(Ticket.updated_at.desc())
                .limit(1)
            ).first()

            _upsert_coverage(
                session,
                owner_id=self.owner_id,
                scope_type="project",
                scope_id=project_id,
                provider="linear",
                earliest_at=_to_utc(earliest_ticket),
                latest_at=_to_utc(latest_ticket),
                is_complete=bool(should_full and not truncated),
                last_run_id=self.run_id,
                notes=("tickets_truncated" if truncated else None),
            )
            run = session.get(SyncRun, self.run_id)
            if run:
                _set_run_status(
                    session,
                    run=run,
                    status="running",
                    message=f"Linear sync done. Fetched {total_fetched} tickets.",
                    progress_current=total_fetched,
                    progress_total=effective_max_tickets,
                )
            session.commit()

    def _persist_linear_page(self, *, project_id: int, issues: list[LinearIssue]) -> int:
        with Session(engine) as session:
            processed = 0
            for issue in issues:
                fingerprint = _ticket_fingerprint(issue)
                created_at = _parse_iso_utc(issue.createdAt) or datetime.now(timezone.utc)
                updated_at = _parse_iso_utc(issue.updatedAt) or created_at
                completed_at = _parse_iso_utc(issue.completedAt)

                ticket_db = session.exec(select(Ticket).where(Ticket.source_id == issue.id)).first()
                if not ticket_db:
                    ticket_db = Ticket(
                        project_id=project_id,
                        source_platform="linear",
                        source_id=issue.id,
                        key=issue.identifier,
                        title=issue.title or "Untitled",
                        description=issue.description,
                        story_points=int(issue.estimate or 0),
                        status_label=issue.state,
                        created_at=created_at,
                        updated_at=updated_at,
                        completed_at=completed_at,
                        url=issue.url,
                        raw_payload={
                            "assignee": issue.assignee,
                            "status_type": issue.state_type,
                            "cycle_name": issue.cycle_name,
                            "cycle_number": issue.cycle_number,
                            "cycle_startsAt": issue.cycle_startsAt,
                            "cycle_endsAt": issue.cycle_endsAt,
                            "fingerprint": fingerprint,
                        },
                    )
                    session.add(ticket_db)
                    session.flush()
                    session.add(
                        TicketEvent(
                            ticket_id=ticket_db.id,
                            event_type="created",
                            to_value=issue.state,
                            payload={"key": issue.identifier},
                        )
                    )
                    processed += 1
                    continue

                stored_fingerprint = (ticket_db.raw_payload or {}).get("fingerprint")
                if stored_fingerprint and stored_fingerprint == fingerprint:
                    continue

                if ticket_db.project_id != project_id:
                    ticket_db.project_id = project_id

                if issue.identifier and issue.identifier != ticket_db.key:
                    session.add(
                        TicketEvent(
                            ticket_id=ticket_db.id,
                            event_type="key_changed",
                            from_value=ticket_db.key,
                            to_value=issue.identifier,
                        )
                    )
                    ticket_db.key = issue.identifier

                if issue.state != ticket_db.status_label:
                    session.add(
                        TicketEvent(
                            ticket_id=ticket_db.id,
                            event_type="status_changed",
                            from_value=ticket_db.status_label,
                            to_value=issue.state,
                        )
                    )
                    ticket_db.status_label = issue.state

                incoming_points = int(issue.estimate or 0)
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

                if issue.title and issue.title != ticket_db.title:
                    session.add(
                        TicketEvent(
                            ticket_id=ticket_db.id,
                            event_type="title_changed",
                            from_value=ticket_db.title,
                            to_value=issue.title,
                        )
                    )
                    ticket_db.title = issue.title

                current_assignee = (ticket_db.raw_payload or {}).get("assignee")
                if issue.assignee != current_assignee:
                    session.add(
                        TicketEvent(
                            ticket_id=ticket_db.id,
                            event_type="assignee_changed",
                            from_value=str(current_assignee),
                            to_value=str(issue.assignee),
                        )
                    )

                if completed_at and ticket_db.completed_at != completed_at:
                    session.add(
                        TicketEvent(
                            ticket_id=ticket_db.id,
                            event_type="completed_at_changed",
                            from_value=str(ticket_db.completed_at),
                            to_value=str(completed_at),
                        )
                    )
                ticket_db.completed_at = completed_at or ticket_db.completed_at

                ticket_db.description = issue.description
                ticket_db.updated_at = updated_at
                ticket_db.url = issue.url or ticket_db.url
                ticket_db.raw_payload = {
                    **(ticket_db.raw_payload or {}),
                    "assignee": issue.assignee,
                    "status_type": issue.state_type,
                    "cycle_name": issue.cycle_name,
                    "cycle_number": issue.cycle_number,
                    "cycle_startsAt": issue.cycle_startsAt,
                    "cycle_endsAt": issue.cycle_endsAt,
                    "fingerprint": fingerprint,
                }

                processed += 1

            session.commit()
            return processed
