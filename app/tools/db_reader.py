from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Optional

from langchain_core.tools import tool
from sqlmodel import Session, select, col
from sqlalchemy import func
from app.core.database.models import (
    Activity,
    AnalysisReport,
    DataCoverage,
    Project,
    ProjectRepository,
    Repository,
    Ticket,
    TicketEvent,
)
from app.core.database.session import engine

_TICKET_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


def _normalize_person_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _rank_candidates(query: str, candidates: list[str], *, limit: int = 6) -> list[str]:
    """
    Rank candidates by fuzzy similarity, with a bias towards substring matches.
    """
    q = _normalize_person_key(query)
    scored: list[tuple[float, str]] = []
    for candidate in candidates:
        c = _normalize_person_key(candidate)
        if not q or not c:
            continue
        if q == c:
            score = 1.0
        elif q in c or c in q:
            score = 0.92
        else:
            score = SequenceMatcher(None, q, c).ratio()
        if score >= 0.62:
            scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in scored[:limit]]


def _resolve_unique_candidate(query: str, candidates: list[str]) -> tuple[Optional[str], list[str]]:
    """
    Returns (resolved, suggestions). If resolved is None and suggestions has many values,
    the caller should ask the user to disambiguate.
    """
    if not candidates:
        return None, []

    query_key = _normalize_person_key(query)
    exact = [c for c in candidates if _normalize_person_key(c) == query_key]
    if len(exact) == 1:
        return exact[0], exact

    ranked = _rank_candidates(query, candidates, limit=6)
    if len(ranked) == 1:
        return ranked[0], ranked

    if ranked:
        # If the top candidate is much better, auto-pick it.
        top = ranked[0]
        top_score = SequenceMatcher(None, _normalize_person_key(query), _normalize_person_key(top)).ratio()
        second = ranked[1] if len(ranked) > 1 else None
        second_score = (
            SequenceMatcher(None, _normalize_person_key(query), _normalize_person_key(second)).ratio()
            if second
            else 0.0
        )
        if top_score >= 0.9 and (top_score - second_score) >= 0.15:
            return top, ranked

    return None, ranked


def _sprint_window(
    *,
    sprint_number: int,
    sprint_length_days: int,
) -> tuple[datetime, datetime]:
    """
    Defines a sprint window as a rolling fixed-length period.

    sprint_number=1 -> current sprint (last N days)
    sprint_number=2 -> previous sprint, etc.
    """
    now = datetime.now(timezone.utc)
    if sprint_number < 1:
        sprint_number = 1

    end = now - timedelta(days=(sprint_number - 1) * sprint_length_days)
    start = end - timedelta(days=sprint_length_days)
    return start, end


def _parse_iso_datetime_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_project_sprint_window(
    *,
    session: Session,
    project_id: int,
    sprint_number: int,
    sprint_length_days: int,
) -> tuple[datetime, datetime, str]:
    """
    Resolve a sprint window using Linear cycle metadata when available.

    If cycle info is missing, fall back to rolling fixed-length windows.
    """
    now = datetime.now(timezone.utc)
    if sprint_number < 1:
        sprint_number = 1

    tickets = session.exec(
        select(Ticket)
        .where(Ticket.project_id == project_id)
        .where(Ticket.source_platform == "linear")
        .limit(5000)
    ).all()

    cycles_by_key: dict[tuple[Optional[int], str, str], tuple[datetime, datetime, str]] = {}
    for t in tickets:
        payload = t.raw_payload or {}
        start = _parse_iso_datetime_utc(payload.get("cycle_startsAt"))
        end = _parse_iso_datetime_utc(payload.get("cycle_endsAt"))
        if not start or not end:
            continue

        name = str(payload.get("cycle_name") or "").strip()
        number = payload.get("cycle_number")
        label = name or (f"Ciclo {number}" if number is not None else "Ciclo")
        key = (number, start.isoformat(), end.isoformat())
        cycles_by_key[key] = (start, end, label)

    if cycles_by_key:
        cycles = sorted(cycles_by_key.values(), key=lambda item: item[0])
        current_index = None
        for idx, (start, end, _) in enumerate(cycles):
            if start <= now < end:
                current_index = idx
                break
        if current_index is None:
            current_index = len(cycles) - 1

        target_index = current_index - (sprint_number - 1)
        if 0 <= target_index < len(cycles):
            start, end, label = cycles[target_index]
            return start, end, f"{label} ({start.date().isoformat()} -> {end.date().isoformat()})"

    start, end = _sprint_window(sprint_number=sprint_number, sprint_length_days=sprint_length_days)
    return start, end, f"Ventana {start.date().isoformat()} -> {end.date().isoformat()}"


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
        if not repo:
            return "No encontré el repositorio en la base de datos."

        report = session.exec(
            select(AnalysisReport)
            .where(AnalysisReport.repository_id == repo.id)
            .order_by(AnalysisReport.id.desc())
        ).first()
        
        if not report:
            return "No encontré reportes de auditoría para ese repositorio."
        return (
            f"DEV: {report.developer_name}\n"
            f"SCORE: {report.quality_score}\n"
            f"SUMMARY:\n{report.feedback_summary}"
        )

@tool
def get_developer_activity(repo_name: str, developer_name: str, limit: int = 7):
    """
    Fetches recent code activity for a specific developer in a repository.
    Use this when the user asks "What did Fede do?" or "Show me the last commits of X".
    """
    with Session(engine) as session:
        # 1. Find Repo
        repo = session.exec(select(Repository).where(Repository.name.contains(repo_name))).first()
        if not repo:
            return "No encontré el repositorio en la base de datos."

        # 1.5 Resolve developer ambiguity (repo-scoped)
        author_rows = session.exec(
            select(Activity.author, func.max(Activity.timestamp).label("last_seen"))
            .where(Activity.repository_id == repo.id)
            .where(Activity.type == "COMMIT")
            .group_by(Activity.author)
            .order_by(func.max(Activity.timestamp).desc())
            .limit(250)
        ).all()
        known_authors = [author for author, _ in author_rows if author]
        resolved_author, suggestions = _resolve_unique_candidate(developer_name, known_authors)
        if resolved_author is None and len(suggestions) > 1:
            return (
                "Encontré más de una persona que coincide con ese nombre. "
                f"¿A cuál te referís? Opciones: {', '.join(suggestions)}"
            )
        resolved_author = resolved_author or (suggestions[0] if suggestions else None)
        if not resolved_author:
            return f"No encontré commits para '{developer_name}' en '{repo.name}'."
        
        # 2. Find commit activities (case-insensitive author search).
        statement = (
            select(Activity)
            .where(Activity.repository_id == repo.id)
            .where(func.lower(Activity.author) == resolved_author.lower())
            .where(Activity.type == "COMMIT")
            .order_by(Activity.timestamp.desc())
            .limit(max(limit * 5, limit))
        )
        activities = session.exec(statement).all()
        
        commits = [a for a in activities if not _is_merge_commit(a.title or "")][:limit]
        if not commits:
            return f"No encontré commits para '{resolved_author}' en '{repo.name}'."
            
        # 3. Format for LLM
        output = f"Últimos {len(commits)} commits de {resolved_author} en {repo.name}:\n"
        for act in commits:
            sha_short = (act.source_id or "")[:7]
            output += f"- {act.timestamp.date()}: {_format_commit_title(act.title)} (sha: {sha_short})\n"
            
        return output
            

@tool
def get_repository_activity(repo_name: str, limit: int = 10):
    """
    Fetches the global recent activity (commits) for a repository, regardless of the author.
    Use this when the user asks "What is the latest commit in the repo?" or "Show me recent activity" without specifying a person.
    """
    with Session(engine) as session:
        repo = session.exec(select(Repository).where(Repository.name.contains(repo_name))).first()
        if not repo:
            return "No encontré el repositorio en la base de datos."
        
        statement = (
            select(Activity)
            .where(Activity.repository_id == repo.id)
            .where(Activity.type == "COMMIT")
            .order_by(Activity.timestamp.desc())
            .limit(limit)
        )
        activities = session.exec(statement).all()
        
        commits = [a for a in activities if not _is_merge_commit(a.title or "")]
        if not commits:
            return f"No encontré commits en '{repo_name}'."
            
        output = f"Últimos {len(commits)} commits en {repo.name} (todos los autores):\n"
        for act in commits:
            sha_short = (act.source_id or "")[:7]
            output += f"- {act.timestamp.date()} | {act.author}: {_format_commit_title(act.title)} (sha: {sha_short})\n"
            
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
            return "No encontré el proyecto en la base de datos."

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No hay repositorios vinculados a este proyecto."

        author_rows = session.exec(
            select(Activity.author, func.max(Activity.timestamp).label("last_seen"))
            .where(Activity.repository_id.in_(repo_ids))
            .where(Activity.type == "COMMIT")
            .group_by(Activity.author)
            .order_by(func.max(Activity.timestamp).desc())
            .limit(250)
        ).all()
        known_authors = [author for author, _ in author_rows if author]
        resolved_author, suggestions = _resolve_unique_candidate(developer_name, known_authors)
        if resolved_author is None and len(suggestions) > 1:
            return (
                "Encontré más de una persona que coincide con ese nombre. "
                f"¿A cuál te referís? Opciones: {', '.join(suggestions)}"
            )
        resolved_author = resolved_author or (suggestions[0] if suggestions else None)
        if not resolved_author:
            return f"No encontré actividad para '{developer_name}' en el proyecto '{project.name}'."

        statement = (
            select(Activity)
            .where(Activity.repository_id.in_(repo_ids))
            .where(func.lower(Activity.author) == resolved_author.lower())
            .where(Activity.type == "COMMIT")
            .order_by(Activity.timestamp.desc())
            .limit(max(limit * 5, limit))
        )
        rows = session.exec(statement).all()
        commits = [a for a in rows if not _is_merge_commit(a.title or "")][:limit]

        if not commits:
            return f"No encontré commits para '{resolved_author}' en el proyecto '{project.name}'."

        repos = session.exec(select(Repository).where(Repository.id.in_(repo_ids))).all()
        repo_name_by_id = {r.id: r.name for r in repos}

        output = f"Últimos {len(commits)} commits de {resolved_author} en el proyecto {project.name}:\n"
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
            return "No encontré el proyecto en la base de datos."

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No hay repositorios vinculados a este proyecto."

        author_rows = session.exec(
            select(Activity.author, func.max(Activity.timestamp).label("last_seen"))
            .where(Activity.repository_id.in_(repo_ids))
            .where(Activity.type == "COMMIT")
            .group_by(Activity.author)
            .order_by(func.max(Activity.timestamp).desc())
            .limit(250)
        ).all()
        known_authors = [author for author, _ in author_rows if author]
        resolved_author, suggestions = _resolve_unique_candidate(developer_name, known_authors)
        if resolved_author is None and len(suggestions) > 1:
            return (
                "Encontré más de una persona que coincide con ese nombre. "
                f"¿A cuál te referís? Opciones: {', '.join(suggestions)}"
            )
        resolved_author = resolved_author or (suggestions[0] if suggestions else None)
        if not resolved_author:
            return f"No encontré commits para '{developer_name}' en el proyecto '{project.name}'."

        statement = (
            select(Activity)
            .where(Activity.repository_id.in_(repo_ids))
            .where(func.lower(Activity.author) == resolved_author.lower())
            .where(Activity.type == "COMMIT")
            .order_by(Activity.timestamp.desc())
            .limit(max(commit_limit * 8, commit_limit))
        )
        rows = session.exec(statement).all()

        commits = [a for a in rows if not _is_merge_commit(a.title or "")][:commit_limit]
        if not commits:
            return f"No encontré commits para '{resolved_author}' en el proyecto '{project.name}'."

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
            f"Commits recientes de {resolved_author} en el proyecto {project.name}:",
            *commit_lines,
            "",
            "Tickets relacionados (por key detectada en commits):",
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
            return "No encontré el proyecto en la base de datos."

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No hay repositorios vinculados a este proyecto."

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
            return f"No encontré desarrolladores para el proyecto '{project.name}'."

        lines = [f"Desarrolladores detectados en el proyecto {project.name} (por commits):"]
        for author, commit_count, last_seen in rows:
            last_label = last_seen.date().isoformat() if last_seen else "unknown"
            lines.append(f"- {author}: {commit_count} commits (last: {last_label})")

        return "\n".join(lines)


@tool
def resolve_developer_candidates(project_name: str, query: str, limit: int = 8):
    """
    Finds likely developer name matches for a user-provided query.

    Use this when the user provides an ambiguous or partial name (e.g., "Fede").
    """
    query = (query or "").strip()
    if not query:
        return "Necesito un nombre para buscar coincidencias."

    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "No encontré el proyecto en la base de datos."

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]

        rows = []
        if repo_ids:
            rows = session.exec(
                select(Activity.author, func.max(Activity.timestamp).label("last_seen"))
                .where(Activity.repository_id.in_(repo_ids))
                .where(Activity.type == "COMMIT")
                .group_by(Activity.author)
                .order_by(func.max(Activity.timestamp).desc())
                .limit(250)
            ).all()

        ticket_rows = session.exec(
            select(Ticket)
            .where(Ticket.project_id == project.id)
            .where(Ticket.source_platform == "linear")
            .order_by(Ticket.updated_at.desc())
            .limit(2500)
        ).all()
        assignees: list[str] = []
        for t in ticket_rows:
            name = str((t.raw_payload or {}).get("assignee") or "").strip()
            if name:
                assignees.append(name)

        authors = [a for a, _ in rows if a]
        candidates = sorted(set(authors + assignees))

        suggestions = _rank_candidates(query, candidates, limit=limit)
        if not suggestions:
            recent = [a for a, _ in rows[: min(10, len(rows))] if a]
            recent_assignees = list(dict.fromkeys(assignees))[: min(10, len(set(assignees)))]
            if recent or recent_assignees:
                chunks = []
                if recent:
                    chunks.append(f"Commits: {', '.join(recent)}")
                if recent_assignees:
                    chunks.append(f"Tickets: {', '.join(recent_assignees)}")
                return (
                    f"No encontré coincidencias para '{query}'. "
                    "Algunas personas vistas recientemente: "
                    + " | ".join(chunks)
                )
            return f"No encontré coincidencias para '{query}'."

        return f"Posibles coincidencias para '{query}': {', '.join(suggestions)}"


@tool
def get_project_sprints_overview(project_name: str):
    """
    Returns the number of sprints (Linear cycles) detected and ticket counts per sprint.

    This tool uses Linear cycle metadata when available; otherwise it will instruct the user
    to provide a sprint length (days) to approximate windows.
    """
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "No encontré el proyecto en la base de datos."

        tickets = session.exec(
            select(Ticket)
            .where(Ticket.project_id == project.id)
            .where(Ticket.source_platform == "linear")
            .order_by(Ticket.updated_at.desc())
            .limit(5000)
        ).all()
        if not tickets:
            return (
                "No tengo tickets en la base de datos para ese proyecto. "
                "Si querés, podés sincronizar Linear desde Connections."
            )

        cycles: dict[tuple[Optional[int], str, str], dict[str, object]] = {}
        no_cycle = 0

        def _cycle_key_from_ticket(t: Ticket):
            payload = t.raw_payload or {}
            start = _parse_iso_datetime_utc(payload.get("cycle_startsAt"))
            end = _parse_iso_datetime_utc(payload.get("cycle_endsAt"))
            if not start or not end:
                return None
            number = payload.get("cycle_number")
            key = (number, start.isoformat(), end.isoformat())
            name = str(payload.get("cycle_name") or "").strip()
            label = name or (f"Ciclo {number}" if number is not None else "Ciclo")
            return key, start, end, label

        for t in tickets:
            info = _cycle_key_from_ticket(t)
            if not info:
                no_cycle += 1
                continue
            key, start, end, label = info
            cycles.setdefault(key, {"start": start, "end": end, "label": label})

        if not cycles:
            return (
                "No encontré información de sprints/ciclos en los tickets (Linear cycle metadata). "
                "Decime cuántos días dura un sprint (ej: 14) y lo calculo por ventanas."
            )

        ordered = sorted(cycles.items(), key=lambda item: item[1]["start"])  # type: ignore[index]

        stats: dict[tuple[Optional[int], str, str], dict[str, int]] = {}
        for key, _meta in ordered:
            stats[key] = {"total": 0, "done": 0, "points_total": 0, "points_done": 0}

        for t in tickets:
            info = _cycle_key_from_ticket(t)
            if not info:
                continue
            key, _start, _end, _label = info
            item = stats.get(key)
            if not item:
                continue
            points = int(t.story_points or 0)
            item["total"] += 1
            item["points_total"] += points
            status_type = str((t.raw_payload or {}).get("status_type") or "").strip()
            if t.completed_at is not None or status_type == "completed":
                item["done"] += 1
                item["points_done"] += points

        lines = [f"Sprints detectados en Linear para {project.name}: {len(ordered)}"]
        for idx, (key, meta) in enumerate(ordered, start=1):
            s = stats.get(key, {"total": 0, "done": 0, "points_total": 0, "points_done": 0})
            start = meta["start"]
            end = meta["end"]
            label = meta["label"]
            lines.append(
                f"- Sprint {idx}: {label} | {start.date().isoformat()} -> {end.date().isoformat()} | "
                f"tickets={s['total']} (done={s['done']}) | points={s['points_total']} (done={s['points_done']})"
            )

        if no_cycle:
            lines.append(f"- Tickets sin sprint/ciclo asignado: {no_cycle}")

        return "\n".join(lines)


@tool
def get_developer_performance_summary(
    project_name: str,
    developer_name: str,
    sprint_number: int = 1,
    sprint_length_days: int = 14,
):
    """
    Summarizes a single developer's performance in a sprint window (commits, PR merges, tickets done).
    Uses Linear cycles for sprint windows when available.
    """
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "No encontré el proyecto en la base de datos."

        start, end, window_label = _resolve_project_sprint_window(
            session=session,
            project_id=project.id or 0,
            sprint_number=sprint_number,
            sprint_length_days=sprint_length_days,
        )

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]

        commit_authors: list[str] = []
        if repo_ids:
            author_rows = session.exec(
                select(Activity.author, func.max(Activity.timestamp).label("last_seen"))
                .where(Activity.repository_id.in_(repo_ids))
                .where(Activity.type == "COMMIT")
                .group_by(Activity.author)
                .order_by(func.max(Activity.timestamp).desc())
                .limit(250)
            ).all()
            commit_authors = [a for a, _ in author_rows if a]

        ticket_rows = session.exec(
            select(Ticket)
            .where(Ticket.project_id == project.id)
            .where(Ticket.source_platform == "linear")
            .order_by(Ticket.updated_at.desc())
            .limit(2500)
        ).all()
        assignees = [
            str((t.raw_payload or {}).get("assignee") or "").strip()
            for t in ticket_rows
            if str((t.raw_payload or {}).get("assignee") or "").strip()
        ]

        candidates = sorted(set(commit_authors + assignees))
        resolved, suggestions = _resolve_unique_candidate(developer_name, candidates)
        if resolved is None and len(suggestions) > 1:
            return f"Para '{developer_name}' encontré varias opciones: {', '.join(suggestions)}. ¿A cuál te referís?"
        resolved = resolved or (suggestions[0] if suggestions else None)
        if not resolved:
            return f"No encontré a '{developer_name}' en el proyecto '{project.name}'."

        commits_count = 0
        prs_merged = 0
        if repo_ids:
            commit_rows = session.exec(
                select(Activity.title)
                .where(Activity.repository_id.in_(repo_ids))
                .where(Activity.type == "COMMIT")
                .where(func.lower(Activity.author) == resolved.lower())
                .where(Activity.timestamp >= start)
                .where(Activity.timestamp < end)
            ).all()
            commits_count = sum(1 for title in commit_rows if not _is_merge_commit(title or ""))

            prs_merged = session.exec(
                select(func.count(Activity.id))
                .where(Activity.repository_id.in_(repo_ids))
                .where(Activity.type == "PR_MERGE")
                .where(func.lower(Activity.author) == resolved.lower())
                .where(Activity.timestamp >= start)
                .where(Activity.timestamp < end)
            ).first() or 0

        author_key = _normalize_person_key(resolved)
        tickets_done = 0
        points = 0
        done_tickets = session.exec(
            select(Ticket)
            .where(Ticket.project_id == project.id)
            .where(Ticket.source_platform == "linear")
            .where(Ticket.completed_at.is_not(None))
            .where(Ticket.completed_at >= start)
            .where(Ticket.completed_at < end)
        ).all()
        for t in done_tickets:
            assignee = str((t.raw_payload or {}).get("assignee") or "")
            if _normalize_person_key(assignee) == author_key:
                tickets_done += 1
                points += int(t.story_points or 0)

        return (
            f"Performance de {resolved} en sprint {sprint_number} ({window_label}):\n"
            f"- commits={commits_count}\n"
            f"- prs_merged={prs_merged}\n"
            f"- tickets_done={tickets_done}\n"
            f"- points_done={points}"
        )


@tool
def get_project_active_developers(project_name: str, days: int = 14, limit: int = 20):
    """
    Lists developers that were active recently (based on latest commit timestamp).
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(days, 1))

    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "No encontré el proyecto en la base de datos."

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No hay repositorios vinculados a este proyecto."

        rows = session.exec(
            select(
                Activity.author,
                func.max(Activity.timestamp).label("last_seen"),
                func.count(Activity.id).label("commit_count"),
            )
            .where(Activity.repository_id.in_(repo_ids))
            .where(Activity.type == "COMMIT")
            .group_by(Activity.author)
            .order_by(func.max(Activity.timestamp).desc())
            .limit(250)
        ).all()

        active = [(a, last, c) for a, last, c in rows if a and last and last >= cutoff]
        active = active[:limit]
        if not active:
            return f"No encontré actividad en los últimos {days} días."

        lines = [f"Desarrolladores activos en los últimos {days} días:"]
        for author, last_seen, commit_count in active:
            last_label = last_seen.date().isoformat() if last_seen else "unknown"
            lines.append(f"- {author}: {commit_count} commits (último: {last_label})")
        return "\n".join(lines)


@tool
def get_project_ticket_board_overview(project_name: str, limit: int = 500):
    """
    Summarizes the ticket board state for a project from the local DB.
    """
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "No encontré el proyecto en la base de datos."

        tickets = session.exec(
            select(Ticket)
            .where(Ticket.project_id == project.id)
            .where(Ticket.source_platform == "linear")
            .order_by(Ticket.updated_at.desc())
            .limit(limit)
        ).all()

        if not tickets:
            return (
                "No tengo tickets en la base de datos para ese proyecto. "
                "Si querés, podés sincronizar Linear desde Connections."
            )

        by_status_type: dict[str, int] = {}
        by_status_label: dict[str, int] = {}
        by_assignee: dict[str, int] = {}
        done = 0

        for t in tickets:
            status_label = (t.status_label or "unknown").strip()
            status_type = str((t.raw_payload or {}).get("status_type") or "unknown").strip()
            assignee = str((t.raw_payload or {}).get("assignee") or "Unassigned").strip()

            by_status_type[status_type] = by_status_type.get(status_type, 0) + 1
            by_status_label[status_label] = by_status_label.get(status_label, 0) + 1
            by_assignee[assignee] = by_assignee.get(assignee, 0) + 1

            if t.completed_at is not None or status_type == "completed":
                done += 1

        lines = [
            f"Resumen de tickets (Linear) para {project.name}:",
            f"- Total (muestra): {len(tickets)}",
            f"- Completados (estimado): {done}",
            "",
            "Por tipo de estado:",
        ]
        for k, v in sorted(by_status_type.items(), key=lambda item: item[1], reverse=True)[:10]:
            lines.append(f"- {k}: {v}")

        lines.append("")
        lines.append("Top assignees (muestra):")
        for k, v in sorted(by_assignee.items(), key=lambda item: item[1], reverse=True)[:10]:
            lines.append(f"- {k}: {v}")

        return "\n".join(lines)


@tool
def get_ticket_counts_by_sprint(
    project_name: str,
    sprint_number: int = 1,
    sprint_length_days: int = 14,
):
    """
    Counts completed tickets per assignee within a sprint-like time window.
    """
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "No encontré el proyecto en la base de datos."

        start, end, window_label = _resolve_project_sprint_window(
            session=session,
            project_id=project.id or 0,
            sprint_number=sprint_number,
            sprint_length_days=sprint_length_days,
        )

        tickets = session.exec(
            select(Ticket)
            .where(Ticket.project_id == project.id)
            .where(Ticket.source_platform == "linear")
            .where(Ticket.completed_at.is_not(None))
            .where(Ticket.completed_at >= start)
            .where(Ticket.completed_at < end)
        ).all()

        if not tickets:
            return f"No encontré tickets completados en ese sprint (ventana: {window_label})."

        by_assignee: dict[str, dict[str, int]] = {}
        for t in tickets:
            assignee = str((t.raw_payload or {}).get("assignee") or "Unassigned").strip()
            item = by_assignee.setdefault(assignee, {"count": 0, "points": 0})
            item["count"] += 1
            item["points"] += int(t.story_points or 0)

        lines = [f"Tickets completados por persona (sprint {sprint_number}, {window_label}):"]
        for assignee, stats in sorted(by_assignee.items(), key=lambda kv: kv[1]["count"], reverse=True):
            lines.append(f"- {assignee}: {stats['count']} tickets | {stats['points']} puntos")
        return "\n".join(lines)


@tool
def get_team_sprint_summary(
    project_name: str,
    sprint_number: int = 1,
    sprint_length_days: int = 14,
):
    """
    Summarizes team performance in a sprint-like time window (commits, PR merges, tickets).
    """
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "No encontré el proyecto en la base de datos."

        start, end, window_label = _resolve_project_sprint_window(
            session=session,
            project_id=project.id or 0,
            sprint_number=sprint_number,
            sprint_length_days=sprint_length_days,
        )

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No hay repositorios vinculados a este proyecto."

        activities = session.exec(
            select(Activity)
            .where(Activity.repository_id.in_(repo_ids))
            .where(Activity.timestamp >= start)
            .where(Activity.timestamp < end)
            .where(Activity.type.in_(["COMMIT", "PR_MERGE"]))
            .order_by(Activity.timestamp.desc())
        ).all()

        commits_by_author: dict[str, int] = {}
        prs_by_author: dict[str, int] = {}
        for a in activities:
            if a.type == "COMMIT":
                if _is_merge_commit(a.title or ""):
                    continue
                commits_by_author[a.author] = commits_by_author.get(a.author, 0) + 1
            elif a.type == "PR_MERGE":
                prs_by_author[a.author] = prs_by_author.get(a.author, 0) + 1

        tickets = session.exec(
            select(Ticket)
            .where(Ticket.project_id == project.id)
            .where(Ticket.source_platform == "linear")
            .where(Ticket.completed_at.is_not(None))
            .where(Ticket.completed_at >= start)
            .where(Ticket.completed_at < end)
        ).all()

        tickets_by_assignee: dict[str, dict[str, int]] = {}
        for t in tickets:
            assignee = str((t.raw_payload or {}).get("assignee") or "Unassigned").strip()
            item = tickets_by_assignee.setdefault(assignee, {"count": 0, "points": 0})
            item["count"] += 1
            item["points"] += int(t.story_points or 0)

        people_keys = set(commits_by_author) | set(prs_by_author) | set(tickets_by_assignee)
        if not people_keys:
            return f"No encontré actividad en el sprint {sprint_number} ({window_label})."

        lines = [f"Resumen del equipo (sprint {sprint_number}, {window_label}):"]
        for person in sorted(people_keys):
            commits = commits_by_author.get(person, 0)
            prs = prs_by_author.get(person, 0)
            tickets_count = tickets_by_assignee.get(person, {}).get("count", 0)
            points = tickets_by_assignee.get(person, {}).get("points", 0)
            lines.append(f"- {person}: commits={commits} | prs_merged={prs} | tickets_done={tickets_count} | points={points}")

        return "\n".join(lines)


@tool
def compare_developers(
    project_name: str,
    developer_a: str,
    developer_b: str,
    sprint_number: int = 1,
    sprint_length_days: int = 14,
):
    """
    Compares two developers in a sprint-like time window.
    """
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "No encontré el proyecto en la base de datos."

        start, end, window_label = _resolve_project_sprint_window(
            session=session,
            project_id=project.id or 0,
            sprint_number=sprint_number,
            sprint_length_days=sprint_length_days,
        )

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No hay repositorios vinculados a este proyecto."

        author_rows = session.exec(
            select(Activity.author, func.max(Activity.timestamp).label("last_seen"))
            .where(Activity.repository_id.in_(repo_ids))
            .where(Activity.type == "COMMIT")
            .group_by(Activity.author)
            .order_by(func.max(Activity.timestamp).desc())
            .limit(250)
        ).all()
        known_authors = [author for author, _ in author_rows if author]

        resolved_a, suggestions_a = _resolve_unique_candidate(developer_a, known_authors)
        if resolved_a is None and len(suggestions_a) > 1:
            return f"Para '{developer_a}' encontré varias opciones: {', '.join(suggestions_a)}. ¿A cuál te referís?"
        resolved_a = resolved_a or (suggestions_a[0] if suggestions_a else None)
        if not resolved_a:
            return f"No encontré a '{developer_a}' en el proyecto."

        resolved_b, suggestions_b = _resolve_unique_candidate(developer_b, known_authors)
        if resolved_b is None and len(suggestions_b) > 1:
            return f"Para '{developer_b}' encontré varias opciones: {', '.join(suggestions_b)}. ¿A cuál te referís?"
        resolved_b = resolved_b or (suggestions_b[0] if suggestions_b else None)
        if not resolved_b:
            return f"No encontré a '{developer_b}' en el proyecto."

        def _metrics_for(author: str) -> dict[str, int]:
            acts = session.exec(
                select(Activity)
                .where(Activity.repository_id.in_(repo_ids))
                .where(func.lower(Activity.author) == author.lower())
                .where(Activity.timestamp >= start)
                .where(Activity.timestamp < end)
                .where(Activity.type.in_(["COMMIT", "PR_MERGE"]))
            ).all()
            commits = 0
            prs = 0
            for a in acts:
                if a.type == "COMMIT":
                    if _is_merge_commit(a.title or ""):
                        continue
                    commits += 1
                elif a.type == "PR_MERGE":
                    prs += 1

            tickets_done = 0
            points = 0
            tickets = session.exec(
                select(Ticket)
                .where(Ticket.project_id == project.id)
                .where(Ticket.source_platform == "linear")
                .where(Ticket.completed_at.is_not(None))
                .where(Ticket.completed_at >= start)
                .where(Ticket.completed_at < end)
            ).all()
            author_key = _normalize_person_key(author)
            for t in tickets:
                assignee = str((t.raw_payload or {}).get("assignee") or "")
                if _normalize_person_key(assignee) == author_key:
                    tickets_done += 1
                    points += int(t.story_points or 0)

            return {"commits": commits, "prs_merged": prs, "tickets_done": tickets_done, "points": points}

        a_stats = _metrics_for(resolved_a)
        b_stats = _metrics_for(resolved_b)

        lines = [
            f"Comparación en sprint {sprint_number} ({window_label}):",
            f"- {resolved_a}: commits={a_stats['commits']} | prs_merged={a_stats['prs_merged']} | tickets_done={a_stats['tickets_done']} | points={a_stats['points']}",
            f"- {resolved_b}: commits={b_stats['commits']} | prs_merged={b_stats['prs_merged']} | tickets_done={b_stats['tickets_done']} | points={b_stats['points']}",
        ]
        return "\n".join(lines)


@tool
def get_developer_commits_with_diffs(
    project_name: str,
    developer_name: str,
    limit: int = 5,
    max_diff_chars: int = 1200,
):
    """
    Returns recent commits for a developer including a diff snippet.

    Use this when a PM asks for an analysis "incluyendo lo desarrollado" (code changes).
    """
    limit = max(1, min(limit, 25))
    max_diff_chars = max(200, min(max_diff_chars, 4000))

    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "No encontré el proyecto en la base de datos."

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No hay repositorios vinculados a este proyecto."

        author_rows = session.exec(
            select(Activity.author, func.max(Activity.timestamp).label("last_seen"))
            .where(Activity.repository_id.in_(repo_ids))
            .where(Activity.type == "COMMIT")
            .group_by(Activity.author)
            .order_by(func.max(Activity.timestamp).desc())
            .limit(250)
        ).all()
        known_authors = [author for author, _ in author_rows if author]
        resolved_author, suggestions = _resolve_unique_candidate(developer_name, known_authors)
        if resolved_author is None and len(suggestions) > 1:
            return (
                "Encontré más de una persona que coincide con ese nombre. "
                f"¿A cuál te referís? Opciones: {', '.join(suggestions)}"
            )
        resolved_author = resolved_author or (suggestions[0] if suggestions else None)
        if not resolved_author:
            return f"No encontré commits para '{developer_name}' en el proyecto '{project.name}'."

        total_commits = session.exec(
            select(func.count(Activity.id))
            .where(Activity.repository_id.in_(repo_ids))
            .where(Activity.type == "COMMIT")
            .where(func.lower(Activity.author) == resolved_author.lower())
        ).first() or 0

        rows = session.exec(
            select(Activity)
            .where(Activity.repository_id.in_(repo_ids))
            .where(Activity.type == "COMMIT")
            .where(func.lower(Activity.author) == resolved_author.lower())
            .order_by(Activity.timestamp.desc())
            .limit(max(limit * 8, limit))
        ).all()
        commits = [a for a in rows if not _is_merge_commit(a.title or "")][:limit]
        if not commits:
            return f"No encontré commits para '{resolved_author}' en el proyecto '{project.name}'."

        repos = session.exec(select(Repository).where(Repository.id.in_(repo_ids))).all()
        repo_name_by_id = {r.id: r.name for r in repos}

        lines: list[str] = [
            f"Commits recientes de {resolved_author} (proyecto {project.name}): mostrando {len(commits)} de ~{total_commits}",
        ]

        for act in commits:
            repo_name = repo_name_by_id.get(act.repository_id, str(act.repository_id))
            sha_short = (act.source_id or "")[:7]
            title = _format_commit_title(act.title or "")
            diff = (act.content or "")
            snippet = diff[:max_diff_chars].rstrip()
            if len(diff) > len(snippet):
                snippet += "\n... (truncado)"
            lines.append(f"\n- {act.timestamp.date()} | {repo_name} | {title} (sha: {sha_short})")
            lines.append(f"  Diff (snippet):\n{snippet}")

        return "\n".join(lines)


@tool
def get_developer_prs_with_related_tickets(
    project_name: str,
    developer_name: str,
    pr_limit: int = 3,
):
    """
    Retrieves the last merged PRs for a developer and includes related ticket context if available.
    """
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "No encontré el proyecto en la base de datos."

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No hay repositorios vinculados a este proyecto."

        pr_author_rows = session.exec(
            select(Activity.author, func.max(Activity.timestamp).label("last_seen"))
            .where(Activity.repository_id.in_(repo_ids))
            .where(Activity.type == "PR_MERGE")
            .group_by(Activity.author)
            .order_by(func.max(Activity.timestamp).desc())
            .limit(200)
        ).all()
        pr_authors = [a for a, _ in pr_author_rows if a]
        resolved_author, suggestions = _resolve_unique_candidate(developer_name, pr_authors)
        if resolved_author is None and len(suggestions) > 1:
            return (
                "Encontré más de una persona que coincide con ese nombre para PRs. "
                f"¿A cuál te referís? Opciones: {', '.join(suggestions)}"
            )
        resolved_author = resolved_author or (suggestions[0] if suggestions else None)
        if not resolved_author:
            return f"No encontré PRs mergeados para '{developer_name}'."

        prs = session.exec(
            select(Activity)
            .where(Activity.repository_id.in_(repo_ids))
            .where(Activity.type == "PR_MERGE")
            .where(func.lower(Activity.author) == resolved_author.lower())
            .order_by(Activity.timestamp.desc())
            .limit(max(pr_limit * 4, pr_limit))
        ).all()

        if not prs:
            return f"No encontré PRs mergeados para '{resolved_author}'."

        selected = prs[:pr_limit]
        ticket_keys: list[str] = []
        pr_blocks: list[str] = []
        for pr in selected:
            title = pr.title or "Untitled PR"
            keys = _extract_ticket_keys(f"{title}\n{pr.content or ''}")
            ticket_keys.extend(keys)
            keys_label = ", ".join(keys) if keys else "none"
            pr_blocks.append(
                f"- PR: {title} | author: {pr.author} | date: {pr.timestamp.date()} | tickets: {keys_label}\n"
                f"  Diff/Body (snippet): {(pr.content or '')[:800]}"
            )

        unique_keys = sorted(set(ticket_keys))
        ticket_lines: list[str] = []
        if unique_keys:
            rows = session.exec(
                select(Ticket)
                .where(Ticket.project_id == project.id)
                .where(Ticket.key.in_(unique_keys))
                .order_by(Ticket.updated_at.desc())
            ).all()
            found = {t.key for t in rows if t.key}
            for t in rows:
                if not t.key:
                    continue
                status = t.status_label or "unknown"
                assignee = str((t.raw_payload or {}).get("assignee") or "Unassigned")
                points = int(t.story_points or 0)
                desc = (t.description or "").strip().replace("\n", " ")
                if len(desc) > 400:
                    desc = desc[:400].rstrip() + "…"
                desc_label = f" | desc: {desc}" if desc else ""
                ticket_lines.append(
                    f"- {t.key}: {t.title} | status: {status} | points: {points} | assignee: {assignee}{desc_label}"
                )
            missing = [k for k in unique_keys if k not in found]
            if missing:
                ticket_lines.append(f"- Keys no resueltas en DB: {', '.join(missing)}")

        output = [
            f"Últimos {len(selected)} PRs mergeados de {resolved_author} (proyecto {project.name}):",
            *pr_blocks,
            "",
            "Tickets relacionados detectados por key:",
            ("\n".join(ticket_lines) if ticket_lines else "No detecté keys o no tengo tickets en DB."),
        ]
        return "\n".join(output)


@tool
def get_data_coverage(project_name: str, repo_name: Optional[str] = None):
    """
    Returns best-effort data coverage information for GitHub (commits) and Linear (tickets).
    """
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "No encontré el proyecto en la base de datos."

        lines: list[str] = [f"Cobertura de datos (proyecto {project.name}):"]

        linear_cov = session.exec(
            select(DataCoverage)
            .where(DataCoverage.scope_type == "project")
            .where(DataCoverage.scope_id == project.id)
            .where(DataCoverage.provider == "linear")
            .order_by(DataCoverage.updated_at.desc())
        ).first()
        if linear_cov:
            earliest = linear_cov.earliest_at.date().isoformat() if linear_cov.earliest_at else "unknown"
            latest = linear_cov.latest_at.date().isoformat() if linear_cov.latest_at else "unknown"
            complete = "sí" if linear_cov.is_complete else "no"
            lines.append(f"- Linear: earliest={earliest} latest={latest} completo={complete}")
        else:
            lines.append("- Linear: sin información de cobertura (posible DB vacía).")

        if repo_name:
            repo = session.exec(select(Repository).where(Repository.name.contains(repo_name))).first()
            if repo:
                gh_cov = session.exec(
                    select(DataCoverage)
                    .where(DataCoverage.scope_type == "repository")
                    .where(DataCoverage.scope_id == repo.id)
                    .where(DataCoverage.provider == "github")
                    .order_by(DataCoverage.updated_at.desc())
                ).first()
                if gh_cov:
                    earliest = gh_cov.earliest_at.date().isoformat() if gh_cov.earliest_at else "unknown"
                    latest = gh_cov.latest_at.date().isoformat() if gh_cov.latest_at else "unknown"
                    complete = "sí" if gh_cov.is_complete else "no"
                    lines.append(f"- GitHub ({repo.name}): earliest={earliest} latest={latest} completo={complete}")
                else:
                    lines.append(f"- GitHub ({repo.name}): sin información de cobertura (posible DB vacía).")

        return "\n".join(lines)
