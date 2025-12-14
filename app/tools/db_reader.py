import re
from typing import Optional
from langchain_core.tools import tool
from sqlmodel import Session, select, col
from sqlalchemy import func
from app.core.database.models import (
    Activity,
    AnalysisReport,
    Project,
    ProjectRepository,
    Repository,
    Ticket,
    TicketEvent,
)
from app.core.database.session import engine

_TICKET_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


def _extract_ticket_keys(text: str) -> list[str]:
    if not text:
        return []
    return sorted(set(_TICKET_KEY_RE.findall(text)))


def _format_commit_title(title: str) -> str:
    if not title:
        return "Untitled"
    return title.replace("Message:", "", 1).strip()


def _is_merge_commit(title: str) -> bool:
    if not title:
        return False
    normalized = title.strip().lower()
    return normalized.startswith("message: merge") or normalized.startswith("merge ")


@tool
def get_latest_audit_report(repo_name: str):
    """
    Fetches the high-level audit summary and score for a repository.
    Use this for general questions like "How is the project doing?" or "What is the quality score?".
    """
    with Session(engine) as session:
        repo = session.exec(select(Repository).where(Repository.name.contains(repo_name))).first()
        if not repo: return "Repo not found."

        report = session.exec(
            select(AnalysisReport)
            .where(AnalysisReport.repository_id == repo.id)
            .order_by(AnalysisReport.id.desc())
        ).first()
        
        if not report: return "No audit found."
        return f"DEV: {report.developer_name}\nSCORE: {report.quality_score}\nSUMMARY: {report.feedback_summary}"

@tool
def get_developer_activity(repo_name: str, developer_name: str, limit: int = 7):
    """
    Fetches recent code activity for a specific developer in a repository.
    Use this when the user asks "What did Fede do?" or "Show me the last commits of X".
    """
    with Session(engine) as session:
        # 1. Find Repo
        repo = session.exec(select(Repository).where(Repository.name.contains(repo_name))).first()
        if not repo: return "Repository not found."
        
        # 2. Find commit activities (case-insensitive author search).
        statement = (
            select(Activity)
            .where(Activity.repository_id == repo.id)
            .where(col(Activity.author).ilike(f"%{developer_name}%"))
            .where(Activity.type == "COMMIT")
            .order_by(Activity.timestamp.desc())
            .limit(max(limit * 5, limit))
        )
        activities = session.exec(statement).all()
        
        commits = [a for a in activities if not _is_merge_commit(a.title or "")][:limit]
        if not commits:
            return f"No commits found for developer '{developer_name}' in '{repo_name}'."
            
        # 3. Format for LLM
        output = f"Last {len(commits)} commits for {developer_name} in {repo.name}:\n"
        for act in commits:
            sha_short = (act.source_id or "")[:7]
            output += f"- {act.timestamp.date()}: {_format_commit_title(act.title)} (sha: {sha_short})\n"
            
        return output


@tool
def get_developer_code_activity_by_project(project_name: str, developer_name: str, limit: int = 7):
    """
    Fetches GitHub commit activity across all repositories linked to an internal Project.

    Use this when the user asks about a developer and the scope is a project that contains
    multiple repositories.
    """
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "Project not found."

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No repositories linked to this project."

        statement = (
            select(Activity)
            .where(Activity.repository_id.in_(repo_ids))
            .where(col(Activity.author).ilike(f"%{developer_name}%"))
            .where(Activity.type == "COMMIT")
            .order_by(Activity.timestamp.desc())
            .limit(max(limit * 5, limit))
        )
        rows = session.exec(statement).all()
        commits = [a for a in rows if not _is_merge_commit(a.title or "")][:limit]

        if not commits:
            return f"No activities found for developer '{developer_name}' in project '{project_name}'."

        repos = session.exec(select(Repository).where(Repository.id.in_(repo_ids))).all()
        repo_name_by_id = {r.id: r.name for r in repos}

        output = f"Last {len(commits)} commits for {developer_name} in project {project.name}:\n"
        for act in commits:
            repo_name = repo_name_by_id.get(act.repository_id, str(act.repository_id))
            sha_short = (act.source_id or "")[:7]
            output += f"- {act.timestamp.date()}: {_format_commit_title(act.title)} (Repo: {repo_name}, sha: {sha_short})\n"

        return output


@tool
def get_developer_recent_work(project_name: str, developer_name: str, commit_limit: int = 7):
    """
    Fetches recent commits for a developer and correlates them with tickets using ticket keys.

    Use this when the user asks: "What did X do in the last N commits and which tickets did they work on?"

    Notes:
    - Tickets are resolved from the `Ticket` table by matching keys like "TRI-123" extracted from commits.
    - If a key is not present in the DB, it will still be listed as a referenced key.
    """

    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "Project not found."

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No repositories linked to this project."

        statement = (
            select(Activity)
            .where(Activity.repository_id.in_(repo_ids))
            .where(col(Activity.author).ilike(f"%{developer_name}%"))
            .where(Activity.type == "COMMIT")
            .order_by(Activity.timestamp.desc())
            .limit(max(commit_limit * 8, commit_limit))
        )
        rows = session.exec(statement).all()

        commits = [a for a in rows if not _is_merge_commit(a.title or "")][:commit_limit]
        if not commits:
            return f"No commits found for developer '{developer_name}' in project '{project.name}'."

        repos = session.exec(select(Repository).where(Repository.id.in_(repo_ids))).all()
        repo_name_by_id = {r.id: r.name for r in repos}

        referenced_keys: list[str] = []
        commit_lines: list[str] = []
        for act in commits:
            repo_name = repo_name_by_id.get(act.repository_id, str(act.repository_id))
            sha_short = (act.source_id or "")[:7]
            title = _format_commit_title(act.title)

            keys = _extract_ticket_keys(f"{title}\n{act.content or ''}")
            referenced_keys.extend(keys)
            keys_label = ", ".join(keys) if keys else "none"

            commit_lines.append(
                f"- {act.timestamp.date()} | {repo_name} | {title} (sha: {sha_short}) | tickets: {keys_label}"
            )

        unique_keys = sorted(set(referenced_keys))

        ticket_section = "No ticket keys were detected in those commits."
        if unique_keys:
            ticket_rows = session.exec(
                select(Ticket)
                .where(Ticket.project_id == project.id)
                .where(Ticket.key.in_(unique_keys))
                .order_by(Ticket.updated_at.desc())
            ).all()

            ticket_ids = [t.id for t in ticket_rows if t.id is not None]
            last_event_by_ticket: dict[int, TicketEvent] = {}
            if ticket_ids:
                events = session.exec(
                    select(TicketEvent)
                    .where(TicketEvent.ticket_id.in_(ticket_ids))
                    .order_by(TicketEvent.occurred_at.desc())
                ).all()
                for e in events:
                    if e.ticket_id not in last_event_by_ticket:
                        last_event_by_ticket[e.ticket_id] = e

            ticket_lines: list[str] = []
            found_keys = {t.key for t in ticket_rows if t.key}
            missing_keys = [k for k in unique_keys if k not in found_keys]

            for t in ticket_rows:
                if not t.key:
                    continue
                last_event = last_event_by_ticket.get(t.id or 0)
                last_event_label = ""
                if last_event:
                    last_event_label = f" | last_event: {last_event.event_type} ({last_event.occurred_at.date()})"

                points = t.story_points
                status = t.status_label or "unknown"
                ticket_lines.append(
                    f"- {t.key}: {t.title} | status: {status} | points: {points}{last_event_label}"
                )

            if missing_keys:
                ticket_lines.append(f"- Unresolved ticket keys (not found in DB): {', '.join(missing_keys)}")

            ticket_section = "\n".join(ticket_lines) if ticket_lines else ticket_section

        output = [
            f"Recent commits for {developer_name} in project {project.name}:",
            *commit_lines,
            "",
            "Related tickets (by extracted key):",
            ticket_section,
        ]
        return "\n".join(output)


@tool
def get_project_developers(project_name: str, limit: int = 20):
    """
    Lists developers (commit authors) seen in a project, ordered by commit count.

    Use this when the user asks: "Who are the developers working on Project X?"
    """

    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "Project not found."

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No repositories linked to this project."

        statement = (
            select(
                Activity.author,
                func.count(Activity.id).label("commit_count"),
                func.max(Activity.timestamp).label("last_seen"),
            )
            .where(Activity.repository_id.in_(repo_ids))
            .where(Activity.type == "COMMIT")
            .group_by(Activity.author)
            .order_by(func.count(Activity.id).desc())
            .limit(limit)
        )
        rows = session.exec(statement).all()
        if not rows:
            return f"No developers found for project '{project.name}'."

        lines = [f"Top developers for project {project.name} (by commits):"]
        for author, commit_count, last_seen in rows:
            last_label = last_seen.date().isoformat() if last_seen else "unknown"
            lines.append(f"- {author}: {commit_count} commits (last: {last_label})")

        return "\n".join(lines)
