from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

from sqlmodel import Session, select

from app.core.database.models import Project, ProjectOwner, ProjectRepository, Repository, SyncRun
from app.core.database.session import engine
from app.services.sync_orchestrator import SyncOrchestrator


def enqueue_sync_run(
    *,
    owner_id: str,
    project_name: str,
    repo_name: Optional[str],
    providers: list[str],
    full_history: bool,
    max_commits: Optional[int],
    max_prs: Optional[int],
    max_tickets: Optional[int],
) -> SyncRun:
    """
    Create a SyncRun row and start a background thread to execute it.

    This is used both by API endpoints and the chat precheck node.
    """

    providers = [p.strip().lower() for p in providers if p and p.strip()]
    provider_label = "all" if len(providers) > 1 else (providers[0] if providers else "all")

    with Session(engine) as session:
        project = session.exec(
            select(Project)
            .join(ProjectOwner, ProjectOwner.project_id == Project.id)
            .where(Project.name == project_name)
            .where(ProjectOwner.owner_id == owner_id)
        ).first()
        if not project:
            # Fall back to existing-by-name (global uniqueness) and claim if unowned.
            existing_by_name = session.exec(select(Project).where(Project.name == project_name)).first()
            if existing_by_name:
                owner_row = session.get(ProjectOwner, existing_by_name.id)
                if owner_row and owner_row.owner_id != owner_id:
                    raise ValueError("Project name already exists for a different user.")
                project = existing_by_name
            else:
                project = Project(name=project_name, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
                session.add(project)
                session.commit()
                session.refresh(project)

            if not session.get(ProjectOwner, project.id):
                session.add(ProjectOwner(project_id=project.id, owner_id=owner_id))
                session.commit()

        repo = None
        if repo_name:
            repo = session.exec(
                select(Repository)
                .where(Repository.owner_id == owner_id)
                .where(Repository.name == repo_name)
            ).first()
            if not repo:
                repo = Repository(
                    url=f"https://github.com/{repo_name}",
                    name=repo_name,
                    owner_id=owner_id,
                )
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
                session.add(ProjectRepository(project_id=project.id, repository_id=repo.id, is_primary=True))
                session.commit()

        existing = session.exec(
            select(SyncRun)
            .where(SyncRun.owner_id == owner_id)
            .where(SyncRun.status.in_(["queued", "running"]))
            .where(SyncRun.provider == provider_label)
            .where(SyncRun.project_id == project.id)
            .where(SyncRun.repository_id == (repo.id if repo else None))
            .order_by(SyncRun.created_at.desc())
        ).first()
        if existing:
            return existing

        run = SyncRun(
            owner_id=owner_id,
            project_id=project.id,
            repository_id=(repo.id if repo else None),
            provider=provider_label,
            status="queued",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            progress_current=0,
            progress_total=None,
            message="Queued",
            details={
                "providers": providers,
                "full_history": full_history,
                "repo_name": repo_name,
                "project_name": project_name,
                "max_commits": max_commits,
                "max_prs": max_prs,
                "max_tickets": max_tickets,
            },
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        run_id = run.id or 0

    def _worker() -> None:
        SyncOrchestrator(owner_id=owner_id, run_id=run_id).run(
            project_name=project_name,
            repo_name=repo_name,
            providers=providers or ["github", "linear"],
            full_history=full_history,
            max_commits=max_commits,
            max_prs=max_prs,
            max_tickets=max_tickets,
        )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    return run
