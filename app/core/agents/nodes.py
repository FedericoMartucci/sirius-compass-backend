from datetime import datetime
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from app.core.agents.state import GraphState
from app.core.llm_factory import get_llm
from app.core.models.domain import DeveloperReport, ActivityType
from app.core.rag.vector_store import search_similar_rules
from app.core.logger import get_logger

logger = get_logger(__name__)

import re

def _extract_ticket_keys(activities: list) -> set:
    """
    Scans code activities for Ticket IDs (e.g. ENG-123).
    """
    keys = set()
    # Common linear/jira pattern: uppercase letters, hyphen, numbers
    pattern = re.compile(r'\b([A-Z0-9]+-\d+)\b')
    
    for act in activities:
        # Check title
        if act.title:
            found = pattern.findall(act.title)
            keys.update(found)
        # Check content
        if act.content:
            found = pattern.findall(act.content)
            keys.update(found)
            
    return keys

def _build_smart_context(activities: list, target_dev: str = None) -> str:
    """
    Constructs a prompt context that prioritizes:
    1. The specific developer's activities (if target_dev is provided).
    2. Tickets that are explicitly referenced in the code (commits/PRs).
    3. Recent tickets if space permits.
    """
    # 1. Filter by Developer (Fuzzy match) if provided
    if target_dev and target_dev.lower() != "team":
        dev_activities = [a for a in activities if target_dev.lower() in a.author.lower()]
        if not dev_activities:
            logger.warning(f"No activities found for {target_dev}. Falling back to team view.")
            relevant_activities = activities
        else:
            relevant_activities = dev_activities
    else:
        relevant_activities = activities

    # 2. Split by Type
    tickets = [a for a in relevant_activities if a.type == ActivityType.TICKET]
    code_changes = [a for a in relevant_activities if a.type != ActivityType.TICKET]

    # 3. Intelligent Selection
    # Find tickets mentioned in code
    referenced_keys = _extract_ticket_keys(code_changes)
    
    referenced_tickets = [t for t in tickets if t.external_key in referenced_keys]
    other_tickets = [t for t in tickets if t.external_key not in referenced_keys]
    
    # Sort others by date
    other_tickets.sort(key=lambda x: x.timestamp, reverse=True)
    
    # We want ~5 tickets. Prioritize referenced ones.
    selected_tickets = referenced_tickets[:5]
    if len(selected_tickets) < 5:
        data_needed = 5 - len(selected_tickets)
        selected_tickets.extend(other_tickets[:data_needed])

    # Select latest code changes (limit 15)
    code_changes.sort(key=lambda x: x.timestamp, reverse=True)
    selected_code = code_changes[:15]
    
    selected_items = selected_tickets + selected_code
    
    # Sort combined list by date desc to show chronological story
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
    Audit the work of developer: **{developer_name}** in the repository **{repo_name}**.
    If the logs contain 'Linear' or 'Tickets', you MUST compare the Ticket Requirements vs. The Implemented Code (Commits/PRs).
    Focus ONLY on tickets that are relevant to the code changes provided. Ignore tickets that seem unrelated to the repository work.
    
    INTERNAL STANDARDS:
    {retrieved_rules}
    
    INSTRUCTIONS:
    Analyze the activity logs and provide a structured review.
    **IMPORTANT: The content of the fields (focus_area, linear_alignment, etc.) MUST BE IN SPANISH.**
    
    Structure:
    - Focus area (Frontend/Backend/etc)
    - Alignment with Linear tickets
    - Code Quality Score (1-10)
    - Security Assessment (Look for hardcoded secrets, PII, etc. Set alerts=True if found).
    - Details of Risks (If alerts=True, explain exactly what was found. If False, leave empty).
    - Detailed Verdict (Feedback for the developer).
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
    
    class AnalysisSchema(BaseModel):
        focus_area: str = Field(description="Main area of work (e.g. Frontend, Backend). In Spanish.")
        linear_alignment: str = Field(description="Analysis of implementation vs ticket requirements. In Spanish.")
        quality_score: int = Field(description="Score from 1-10 based on code quality and standards")
        security_alerts: bool = Field(description="True if ANY security risks (secrets, PII, unsafe patterns, unsafe code or methodologies) are detected")
        risk_details: Optional[str] = Field(description="If security_alerts is True, explain the specific risks found. In Spanish.")
        verdict: str = Field(description="Detailed feedback and conclusion. In Spanish.")

    try:
        structured_llm = llm.with_structured_output(AnalysisSchema)
        response = structured_llm.invoke(messages)
    except Exception as e:
        logger.error(f"Structured Output Failed: {e}. Falling back to text.")
        # Fallback to normal text generation if structured fails
        response = llm.invoke(messages) # This returns BaseMessage
        return {"analysis_logs": [f"Analysis failed to structure output: {str(e)}"]}

    formatted_log = f"""
    **Enfoque**: {response.focus_area}
    **Linear vs Implementación**: {response.linear_alignment}
    **Calidad de Código**: {response.quality_score}/10
    **Veredicto**: {response.verdict}
    """

    return {
        "analysis_logs": [formatted_log],
        "structured_analysis": response.dict()
    }

def generate_report_node(state: GraphState):
    # (Misma implementación anterior)
    # ...
    developer_name = state["developer_name"]
    analysis_text = "\n".join(state["analysis_logs"])
    activities = state.get("activities", [])
    
    # Logic to extract score from text or default
    structured_data = state.get("structured_analysis", {})
    quality_score = structured_data.get("quality_score", 5)
    
    report = DeveloperReport(
        developer_name=developer_name,
        period_start=datetime.now(),
        period_end=datetime.now(),
        tasks_completed=len([a for a in activities if a.type == ActivityType.TICKET]),
        prs_merged=len([a for a in activities if "MERGE" in str(a.type)]),
        quality_score=quality_score, 
        feedback_summary=analysis_text,
        detected_skills=["Python", "React"] 
    )
    
    # Attach security and risk fields
    if "security_alerts" in structured_data:
        setattr(report, "security_alerts", structured_data["security_alerts"])
    if "risk_details" in structured_data:
        setattr(report, "risk_details", structured_data["risk_details"])
    
    return {"final_report": report}