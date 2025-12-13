# app/core/agents/nodes.py
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.agents.state import GraphState
from app.core.llm_factory import get_llm
from app.core.models.domain import DeveloperReport, ActivityType
from app.core.rag.vector_store import search_similar_rules
from app.core.logger import get_logger

logger = get_logger(__name__)

def _build_smart_context(activities: list, target_dev: str) -> str:
    """
    Constructs a prompt context that prioritizes:
    1. The specific developer's activities.
    2. A mix of Code (Commits/PRs) and Requirements (Tickets).
    """
    # 1. Filter by Developer (Fuzzy match)
    # If target is "Team", we use everything.
    if target_dev and target_dev.lower() != "team":
        dev_activities = [a for a in activities if target_dev.lower() in a.author.lower()]
        # If the specific dev has no activity, fallback to all (but warn in prompt)
        if not dev_activities:
            logger.warning(f"No activities found for {target_dev}. Falling back to team view.")
            relevant_activities = activities
        else:
            relevant_activities = dev_activities
    else:
        relevant_activities = activities

    # 2. Split by Type to ensure Linear isn't drowned out by GitHub
    tickets = [a for a in relevant_activities if a.type == ActivityType.TICKET]
    code_changes = [a for a in relevant_activities if a.type != ActivityType.TICKET]

    # 3. Select items for Context Window (Limit ~20 items total)
    # We want at least 5 tickets if available, rest code.
    selected_items = []
    selected_items.extend(tickets[:5]) 
    selected_items.extend(code_changes[:15])

    # Sort by date desc to show latest logic
    selected_items.sort(key=lambda x: x.timestamp, reverse=True)

    context_text = ""
    for act in selected_items:
        context_text += f"""
        ---
        [Type]: {act.type.value}
        [Platform]: {act.source_platform.upper()}
        [Author]: {act.author}
        [Date]: {act.timestamp}
        [Content]: 
        {act.content[:1500]} 
        ---
        """
    return context_text

def analyze_activities_node(state: GraphState):
    """
    Analyst Node with Smart Context.
    """
    activities = state["activities"]
    repo_name = state["repo_name"]
    developer_name = state["developer_name"]
    
    # Short-circuit if empty
    if not activities:
        return {
            "analysis_logs": ["No activities found. Cannot generate audit."]
        }
    
    logger.info(f"🧠 Agent: Analyzing {len(activities)} raw items for dev: {developer_name}...")
    
    llm = get_llm(temperature=0) 
    
    # 1. RAG
    retrieved_rules = search_similar_rules("security credentials testing language", k=5)

    # 2. SMART CONTEXT
    context_text = _build_smart_context(activities, developer_name)

    # 3. PROMPT
    system_prompt = f"""
    You are Sirius Compass, a Senior Tech Lead.
    
    GOAL:
    Audit the work of developer: **{developer_name}**.
    If the logs contain 'Linear' or 'Tickets', you MUST compare the Ticket Requirements vs. The Implemented Code (Commits/PRs).
    
    INTERNAL STANDARDS:
    {retrieved_rules}
    
    OUTPUT FORMAT (Spanish):
    - **Enfoque**: (Did they work on frontend, backend, or bugs?)
    - **Linear vs Implementación**: (Did the code match the ticket? If no tickets seen, say "No Tickets found for this dev").
    - **Calidad de Código**: (Score 1-10. Check for Tests, Clean Code, Spanglish).
    - **Seguridad**: (Hardcoded secrets? PII?).
    - **Veredicto**: (Detailed feedback for {developer_name}).
    """

    user_prompt = f"""
    Repository: {repo_name}
    Activity Log:
    {context_text}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    return {"analysis_logs": [response.content]}

def generate_report_node(state: GraphState):
    # (Misma implementación anterior)
    # ...
    developer_name = state["developer_name"]
    analysis_text = "\n".join(state["analysis_logs"])
    activities = state.get("activities", [])
    
    # Logic to extract score from text or default
    # For MVP we default to 5, in production we use Structured Output
    
    report = DeveloperReport(
        developer_name=developer_name,
        period_start=datetime.now(),
        period_end=datetime.now(),
        tasks_completed=len([a for a in activities if a.type == ActivityType.TICKET]),
        prs_merged=len([a for a in activities if "MERGE" in str(a.type)]),
        quality_score=5, 
        feedback_summary=analysis_text,
        detected_skills=["Python", "React"] 
    )
    
    return {"final_report": report}