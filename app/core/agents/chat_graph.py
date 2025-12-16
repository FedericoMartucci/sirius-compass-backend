# app/core/agents/chat_graph.py
import json
import re
from datetime import datetime, timezone
from datetime import timedelta
from typing import Any, Optional

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, SystemMessage
from sqlmodel import Session, select
from sqlalchemy import func

from app.core.agents.state import ChatState
from app.core.llm_factory import get_llm
# Import both tools.
from app.tools.db_reader import (
    compare_developers,
    get_developer_activity,
    get_developer_code_activity_by_project,
    get_developer_commits_with_diffs,
    get_developer_performance_summary,
    get_developer_recent_work,
    get_developer_prs_with_related_tickets,
    get_data_coverage,
    get_latest_audit_report,
    get_project_active_developers,
    get_project_developers,
    get_project_sprints_overview,
    get_project_ticket_board_overview,
    get_repository_activity,
    get_team_sprint_summary,
    get_ticket_counts_by_sprint,
    resolve_developer_candidates,
)
from app.core.database.models import (
    Activity,
    DataCoverage,
    IntegrationCredential,
    Project,
    ProjectOwner,
    ProjectIntegration,
    ProjectRepository,
    Repository,
    Ticket,
)
from app.core.database.session import engine
from app.services.sync_queue import enqueue_sync_run

def build_chat_graph():
    """
    Interactive Chatbot with Granular DB Access.
    """
    # 1. Setup tools (includes get_developer_activity).
    tools = [
        get_latest_audit_report,
        get_developer_activity,
        get_developer_code_activity_by_project,
        get_developer_recent_work,
        get_developer_commits_with_diffs,
        get_project_developers,
        get_project_active_developers,
        resolve_developer_candidates,
        compare_developers,
        get_ticket_counts_by_sprint,
        get_team_sprint_summary,
        get_project_sprints_overview,
        get_developer_performance_summary,
        get_project_ticket_board_overview,
        get_developer_prs_with_related_tickets,
        get_data_coverage,
        get_repository_activity,
    ]
    
    llm = get_llm(temperature=0, streaming=True).bind_tools(tools)

    _sync_request_re = re.compile(r"<sync_request>(.*?)</sync_request>", re.DOTALL)
    _repo_full_name_re = re.compile(r"^[^\s/]+/[^\s/]+$")
    _commit_word_re = re.compile(r"\bcommit(s)?\b", re.IGNORECASE)
    _pr_word_re = re.compile(r"\bpr(s)?\b", re.IGNORECASE)

    def _coerce_message_role(msg: Any) -> Optional[str]:
        if isinstance(msg, tuple) and len(msg) == 2:
            return str(msg[0])
        if hasattr(msg, "type"):
            t = getattr(msg, "type")
            if isinstance(t, str):
                return t
        if hasattr(msg, "role"):
            r = getattr(msg, "role")
            if isinstance(r, str):
                return r
        if msg.__class__.__name__.lower().endswith("message"):
            name = msg.__class__.__name__.lower()
            if "human" in name:
                return "user"
            if "ai" in name:
                return "assistant"
            if "system" in name:
                return "system"
        return None

    def _coerce_message_text(msg: Any) -> str:
        if isinstance(msg, tuple) and len(msg) == 2:
            return str(msg[1] or "")
        if hasattr(msg, "content"):
            return str(getattr(msg, "content") or "")
        return str(msg or "")

    def _find_last_message(messages: list[Any], role: str) -> Optional[Any]:
        for msg in reversed(messages):
            if _coerce_message_role(msg) == role:
                return msg
        return None

    def _is_affirmative(text: str) -> bool:
        normalized = text.strip().lower()
        return normalized in {
            "si",
            "sí",
            "dale",
            "ok",
            "okay",
            "de acuerdo",
            "confirmo",
            "confirmar",
            "acepto",
            "yes",
            "y",
        }

    def _is_negative(text: str) -> bool:
        normalized = text.strip().lower()
        return normalized in {"no", "nop", "nah", "cancelar", "no gracias"}

    def _extract_sync_request(text: str) -> Optional[dict[str, Any]]:
        match = _sync_request_re.search(text or "")
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except Exception:
            return None

    def _needs_github_full_history(user_text: str) -> bool:
        t = user_text.lower()
        return any(
            phrase in t
            for phrase in (
                "todos los commits",
                "all commits",
                "historial completo",
                "hasta hoy",
                "hasta el dia de hoy",
                "hasta el día de hoy",
            )
        )

    def _normalize_repo_name(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        value = str(raw).strip()
        if not value or " " in value:
            return None
        if _repo_full_name_re.match(value):
            return value
        return None

    def _mentions_github_work(user_text: str) -> bool:
        t = user_text.strip()
        if not t:
            return False
        # Use word boundaries to avoid false positives like "proyecto" -> "pr".
        if _commit_word_re.search(t) or _pr_word_re.search(t):
            return True
        lowered = t.lower()
        return any(
            phrase in lowered
            for phrase in (
                "pull request",
                "merge request",
                "github",
                "repositorio",
                "repository",
            )
        )

    def _mentions_performance_request(user_text: str) -> bool:
        t = (user_text or "").lower()
        return any(
            k in t
            for k in (
                "compar",
                "compare",
                "rendimiento",
                "performance",
                "perfom",
                "perfomo",
                "velocity",
                "velocidad",
                "kpi",
                "equipo",
                "team",
                "sprint",
            )
        )

    def _needs_linear_history(user_text: str) -> bool:
        t = user_text.lower()
        return any(k in t for k in ("ticket", "tickets", "tablero", "sprint", "ciclo"))

    def _wants_fresh_data(user_text: str) -> bool:
        t = user_text.lower()
        return any(
            k in t
            for k in (
                "latest",
                "último",
                "ultimos",
                "últimos",
                "reciente",
                "recién",
                "ahora",
                "nuevo",
                "nuevos",
                "actualizado",
                "actualiza",
                "actualizá",
            )
        )

    def _should_prompt_sync(
        state: ChatState,
        *,
        user_text_override: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        meta = state.get("meta") or {}
        owner_id = str(meta.get("user_id") or "").strip()
        project_name = str(meta.get("project_name") or "").strip()
        repo_name = _normalize_repo_name(meta.get("repo_name"))
        if not owner_id or not project_name:
            return None

        user_msg = _find_last_message(state.get("messages", []), "user")
        user_text = (user_text_override if user_text_override is not None else _coerce_message_text(user_msg))
        if not user_text:
            return None

        need_github = _needs_github_full_history(user_text) or _mentions_github_work(user_text)
        need_linear = _needs_linear_history(user_text)
        if _mentions_performance_request(user_text) and not need_github and not need_linear:
            need_github = True
            need_linear = True

        providers: list[str] = []
        if need_github:
            providers.append("github")
        if need_linear:
            providers.append("linear")
        if not providers:
            return None

        with Session(engine) as session:
            # Credential checks (best-effort)
            if "github" in providers:
                gh_cred = session.exec(
                    select(IntegrationCredential)
                    .where(IntegrationCredential.owner_id == owner_id)
                    .where(IntegrationCredential.provider == "github")
                    .order_by(IntegrationCredential.updated_at.desc())
                ).first()
                if not gh_cred:
                    return {
                        "type": "missing_credentials",
                        "provider": "github",
                        "message": "No tengo credenciales de GitHub configuradas. Conectá GitHub en Connections para poder sincronizar.",
                    }

            project = session.exec(
                select(Project)
                .join(ProjectOwner, ProjectOwner.project_id == Project.id)
                .where(Project.name == project_name)
                .where(ProjectOwner.owner_id == owner_id)
            ).first()
            if not project:
                return {
                    "type": "missing_project",
                    "message": "No encontré el proyecto en la base de datos. Crealo o seleccioná un proyecto y luego intentá de nuevo.",
                }

            if "github" in providers and not repo_name:
                primary_repo = session.exec(
                    select(Repository)
                    .join(ProjectRepository, ProjectRepository.repository_id == Repository.id)
                    .where(ProjectRepository.project_id == project.id)
                    .where(Repository.owner_id == owner_id)
                    .order_by(ProjectRepository.is_primary.desc(), ProjectRepository.created_at.desc())
                ).first()
                repo_name = primary_repo.name if primary_repo else None

            if "github" in providers and not repo_name:
                return {
                    "type": "missing_repo",
                    "message": "Necesito saber qué repositorio de GitHub usar (formato 'owner/repo'). Seleccionalo en Connections o decímelo explícitamente.",
                }

            if "linear" in providers:
                lin_integration = session.exec(
                    select(ProjectIntegration)
                    .where(ProjectIntegration.project_id == project.id)
                    .where(ProjectIntegration.provider == "linear")
                    .order_by(ProjectIntegration.updated_at.desc())
                ).first()
                if not lin_integration:
                    return {
                        "type": "missing_credentials",
                        "provider": "linear",
                        "message": "No tengo Linear conectado para este proyecto. Conectalo en Connections para poder sincronizar tickets.",
                    }

            full_history = _needs_github_full_history(user_text)
            max_commits: Optional[int] = None if full_history else 300
            max_prs: Optional[int] = None if full_history else 200
            max_tickets: Optional[int] = None if full_history else 200
            wants_fresh = _wants_fresh_data(user_text)
            freshness_cutoff = datetime.now(timezone.utc) - timedelta(hours=6)

            # Coverage checks: propose sync if DB is empty/incomplete/stale for requested providers.
            needs_sync = False
            optional_summary: list[str] = []
            severity = "required"
            if "github" in providers and repo_name:
                repo = session.exec(
                    select(Repository)
                    .where(Repository.owner_id == owner_id)
                    .where(Repository.name == repo_name)
                ).first()
                if repo:
                    activity_count = session.exec(
                        select(func.count(Activity.id)).where(Activity.repository_id == repo.id)
                    ).first()
                    commit_count = session.exec(
                        select(func.count(Activity.id))
                        .where(Activity.repository_id == repo.id)
                        .where(Activity.type == "COMMIT")
                    ).first()
                    pr_merge_count = session.exec(
                        select(func.count(Activity.id))
                        .where(Activity.repository_id == repo.id)
                        .where(Activity.type == "PR_MERGE")
                    ).first()
                    cov = session.exec(
                        select(DataCoverage)
                        .where(DataCoverage.scope_type == "repository")
                        .where(DataCoverage.scope_id == repo.id)
                        .where(DataCoverage.provider == "github")
                        .where(DataCoverage.owner_id == owner_id)
                    ).first()
                    if (activity_count or 0) == 0:
                        needs_sync = True
                    elif full_history and (not cov or not cov.is_complete):
                        needs_sync = True
                        severity = "optional"
                        earliest = cov.earliest_at.date().isoformat() if cov and cov.earliest_at else "unknown"
                        latest = cov.latest_at.date().isoformat() if cov and cov.latest_at else "unknown"
                        optional_summary.append(
                            f"- GitHub ({repo.name}): tengo {commit_count or 0} commits y {pr_merge_count or 0} PRs mergeadas entre {earliest} y {latest}, pero NO es historial completo."
                        )
                    elif wants_fresh and (not cov or cov.updated_at < freshness_cutoff):
                        needs_sync = True
                        severity = "optional"
                        last = cov.updated_at.isoformat() if cov and cov.updated_at else "unknown"
                        optional_summary.append(
                            f"- GitHub ({repo.name}): tengo {commit_count or 0} commits y {pr_merge_count or 0} PRs mergeadas; últimos datos sincronizados ~{last} (posible desactualización)."
                        )
                else:
                    needs_sync = True

            if "linear" in providers:
                ticket_count = session.exec(
                    select(func.count(Ticket.id))
                    .where(Ticket.project_id == project.id)
                    .where(Ticket.source_platform == "linear")
                ).first()
                cov = session.exec(
                    select(DataCoverage)
                    .where(DataCoverage.scope_type == "project")
                    .where(DataCoverage.scope_id == project.id)
                    .where(DataCoverage.provider == "linear")
                    .where(DataCoverage.owner_id == owner_id)
                ).first()
                if (ticket_count or 0) == 0:
                    needs_sync = True
                elif full_history and (not cov or not cov.is_complete):
                    needs_sync = True
                    severity = "optional"
                    earliest = cov.earliest_at.date().isoformat() if cov and cov.earliest_at else "unknown"
                    latest = cov.latest_at.date().isoformat() if cov and cov.latest_at else "unknown"
                    optional_summary.append(
                        f"- Linear (proyecto {project.name}): tengo {ticket_count or 0} tickets entre {earliest} y {latest}, pero NO es historial completo."
                    )
                elif wants_fresh and (not cov or cov.updated_at < freshness_cutoff):
                    needs_sync = True
                    severity = "optional"
                    last = cov.updated_at.isoformat() if cov and cov.updated_at else "unknown"
                    optional_summary.append(
                        f"- Linear (proyecto {project.name}): tengo {ticket_count or 0} tickets; últimos datos sincronizados ~{last} (posible desactualización)."
                    )

            if not needs_sync:
                return None

            return {
                "type": "sync_request",
                "project_name": project_name,
                "repo_name": (repo_name if "github" in providers else None),
                "providers": providers,
                "full_history": full_history,
                "max_commits": max_commits,
                "max_prs": max_prs,
                "max_tickets": max_tickets,
                "severity": severity,
                "summary": "\n".join(optional_summary) if optional_summary else None,
            }

    def precheck_node(state: ChatState):
        meta = state.get("meta") or {}
        owner_id = str(meta.get("user_id") or "").strip()
        project_name = str(meta.get("project_name") or "").strip()
        repo_name = str(meta.get("repo_name") or "").strip() or None

        messages = state.get("messages", [])
        user_msg = _find_last_message(messages, "user")
        user_text = _coerce_message_text(user_msg)
        assistant_msg = _find_last_message(messages[:-1], "assistant") if messages else None
        assistant_text = _coerce_message_text(assistant_msg)

        pending = _extract_sync_request(assistant_text)
        if pending and _is_affirmative(user_text) and owner_id:
            run = enqueue_sync_run(
                owner_id=owner_id,
                project_name=str(pending.get("project_name") or project_name).strip() or project_name,
                repo_name=pending.get("repo_name"),
                providers=list(pending.get("providers") or ["github", "linear"]),
                full_history=bool(pending.get("full_history")),
                max_commits=pending.get("max_commits"),
                max_prs=pending.get("max_prs"),
                max_tickets=pending.get("max_tickets"),
            )
            run_id = run.id or 0
            response = (
                "Perfecto. Inicié la sincronización en segundo plano. "
                f"Podés seguir preguntando mientras se completa. (sync_run_id: {run_id})\n"
                f"<sync_run>{{\"run_id\": {run_id}}}</sync_run>"
            )
            return {"messages": [AIMessage(content=response)], "skip_agent": True}

        # If the assistant asked to sync but did not include a <sync_request> block,
        # accept a simple "sí" and infer the intended sync from the previous user message.
        if (not pending) and _is_affirmative(user_text) and owner_id and assistant_text:
            asked_to_sync = ("sincron" in assistant_text.lower()) or ("sync" in assistant_text.lower())
            if asked_to_sync:
                prev_user_msg = _find_last_message(messages[:-1], "user")
                prev_user_text = _coerce_message_text(prev_user_msg)
                inferred = _should_prompt_sync(state, user_text_override=prev_user_text)
                if inferred and inferred.get("type") == "missing_credentials":
                    return {"messages": [AIMessage(content=str(inferred.get("message") or ""))], "skip_agent": True}
                if inferred and inferred.get("type") == "missing_project":
                    return {"messages": [AIMessage(content=str(inferred.get("message") or ""))], "skip_agent": True}
                if inferred and inferred.get("type") == "missing_repo":
                    return {"messages": [AIMessage(content=str(inferred.get("message") or ""))], "skip_agent": True}
                if inferred and inferred.get("type") == "sync_request":
                    run = enqueue_sync_run(
                        owner_id=owner_id,
                        project_name=str(inferred.get("project_name") or project_name).strip() or project_name,
                        repo_name=inferred.get("repo_name"),
                        providers=list(inferred.get("providers") or ["github", "linear"]),
                        full_history=bool(inferred.get("full_history")),
                        max_commits=inferred.get("max_commits"),
                        max_prs=inferred.get("max_prs"),
                        max_tickets=inferred.get("max_tickets"),
                    )
                    run_id = run.id or 0
                    response = (
                        "Listo. Inicié la sincronización en segundo plano. "
                        f"Podés seguir preguntando mientras se completa. (sync_run_id: {run_id})\n"
                        f"<sync_run>{{\"run_id\": {run_id}}}</sync_run>"
                    )
                    return {"messages": [AIMessage(content=response)], "skip_agent": True}

        if pending and _is_negative(user_text):
            # Continue to agent with whatever is already in the DB.
            return {"skip_agent": False}

        prompt = _should_prompt_sync(state)
        if not prompt:
            return {"skip_agent": False}

        if prompt.get("type") == "missing_credentials":
            return {"messages": [AIMessage(content=str(prompt.get("message") or ""))], "skip_agent": True}

        if prompt.get("type") == "missing_project":
            return {"messages": [AIMessage(content=str(prompt.get("message") or ""))], "skip_agent": True}

        if prompt.get("type") == "missing_repo":
            return {"messages": [AIMessage(content=str(prompt.get("message") or ""))], "skip_agent": True}

        sync_payload = {
            "project_name": prompt.get("project_name"),
            "repo_name": prompt.get("repo_name"),
            "providers": prompt.get("providers"),
            "full_history": prompt.get("full_history"),
            "max_commits": prompt.get("max_commits"),
            "max_prs": prompt.get("max_prs"),
            "max_tickets": prompt.get("max_tickets"),
        }
        summary = prompt.get("summary")
        severity = str(prompt.get("severity") or "required")
        if summary and severity == "optional":
            question = (
                "Puedo responder con lo que tengo, pero la base local parece incompleta/desactualizada:\n"
                f"{summary}\n\n"
                "¿Querés que sincronice ahora (y lo guarde en la base de datos) para responder mejor?\n"
                "Respondé 'sí' para iniciar o 'no' para seguir con lo que tengo.\n"
                f"<sync_request>{json.dumps(sync_payload, ensure_ascii=False)}</sync_request>"
            )
        else:
            question = (
                "No tengo suficientes datos sincronizados localmente para responder con confianza. "
                "¿Querés que sincronice ahora (y lo guarde en la base de datos)?\n"
                "Respondé 'sí' para iniciar o 'no' para seguir con lo que tengo.\n"
                f"<sync_request>{json.dumps(sync_payload, ensure_ascii=False)}</sync_request>"
            )
        return {"messages": [AIMessage(content=question)], "skip_agent": True}

    def _route_after_precheck(state: ChatState):
        if state.get("skip_agent"):
            return END
        return "agent"

    # 2. Agent Node
    def agent_node(state: ChatState):
        system_msg = SystemMessage(content="""
        You are Sirius Compass.
        
        CAPABILITIES:
        1. General Audit: Use 'get_latest_audit_report' for scores/summaries.
        2. Developer Commits: Use 'get_developer_activity' for a repo-scoped view (requires developer name).
        3. Project Commits: Use 'get_developer_code_activity_by_project' for project-scoped commits across repos.
        4. Commits + Tickets: Use 'get_developer_recent_work' when the user asks for commits AND related tickets.
        4b. Commit Diffs: Use 'get_developer_commits_with_diffs' when the user asks "incluyendo lo desarrollado" or wants code-level analysis.
        5. Developers List: Use 'get_project_developers' or 'get_project_active_developers' to list who is working on a project.
        6. Name Disambiguation: Use 'resolve_developer_candidates' if the user gives a partial name and you need to ask "which one?".
        7. PM Metrics (Sprint): Use 'get_project_sprints_overview' for "cuántos sprints hubo / tickets por sprint". Use 'get_ticket_counts_by_sprint' and 'get_team_sprint_summary' for sprint-window metrics.
        7b. Developer Performance: Use 'get_developer_performance_summary' for "cómo performó X".
        8. Compare Devs: Use 'compare_developers' when asked "compare X vs Y".
        9. Tickets Board: Use 'get_project_ticket_board_overview' for board questions.
        10. PR + Ticket Context: Use 'get_developer_prs_with_related_tickets' for "últimas PRs vs tickets".
        11. Coverage: Use 'get_data_coverage' if the user asks for "all history" or "hasta hoy".
        12. Global Activity: Use 'get_repository_activity' to see the latest commits from ANYONE in a repo. Use this if no specific developer is mentioned.

        IMPORTANT:
        - If the user asks "which tickets" (or similar), do NOT guess from branch names. Use 'get_developer_recent_work'.
        - If a developer name is ambiguous, ask a clarification question (or use 'resolve_developer_candidates') before answering.
        - If the user asks for "all commits / hasta hoy" and coverage is not complete, explain what you have and ask if they want to sync more data.
        - Never say "I don't have a tool". If data is missing, ask the user if they want to sync (or suggest Connections).
        
        Try to answer the user's question directly using the data from the tools.

        Always respond in Spanish.
        """)
        
        return {"messages": [llm.invoke([system_msg] + state["messages"])]}

    # Graph wiring.
    workflow = StateGraph(ChatState)
    workflow.add_node("precheck", precheck_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("precheck")
    workflow.add_conditional_edges("precheck", _route_after_precheck)
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()
