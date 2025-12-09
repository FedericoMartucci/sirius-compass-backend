from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from app.core.agents.state import GraphState
from app.core.llm_factory import get_llm
from app.core.models.domain import DeveloperReport

def analyze_activities_node(state: GraphState):
    """
    Analyst Node: it takes raw activities (with code diffs) and asks the LLM to find patterns.
    """
    print("Analyzing code patterns and diffs...")
    
    activities = state["activities"]
    repo_name = state["repo_name"]
    # We use a low temperature for objective technical analysis
    llm = get_llm(temperature=0.2) 
    
    # 1. Prepare Context (Summarize Diffs to fit context window efficiently)
    # We prioritize the latest 15 activities for the MVP
    context_text = ""
    for act in activities[:15]:
        context_text += f"""
        ---
        [Activity Type]: {act.type.value}
        [Date]: {act.timestamp}
        [Content/Diff Summary]: 
        {act.content[:1500]} 
        ---
        """

    # 2. Define the Prompt (Strict Auditor Persona)
    system_prompt = """
    You are Sirius Compass, an uncompromising Senior Tech Lead and Code Auditor.
    Your goal is to perform a CRITICAL code review. Do not sugarcoat your findings.
    
    You must evaluate the developer's work against the "Sirius Engineering Standards":
    
    1. LANGUAGE CONSISTENCY: The codebase (variables, functions, comments) MUST be 100% in English. 
       - If you detect Spanish comments or Spanglish, FLAG IT immediately as a critical issue.
       
    2. REPOSITORY HYGIENE: 
       - Committing test files (e.g., 'test_models.py') inside the main application logic is forbidden.
       - Temporary files or junk code must not be committed.
       
    3. ATOMICITY: Commits should solve one thing. Giant commits or vague messages like "changes" are bad practices.
    
    OUTPUT FORMAT:
    Provide a technical report in Spanish.
    - Start with a "Calidad Técnica" score (1-10) based on your strict assessment.
    - List "Puntos Fuertes" (briefly).
    - List "Áreas de Mejora" (be very specific about files and lines if possible).
    - End with a "Veredicto" (Junior, Semi-Senior, Senior) based solely on the evidence.
    """

    user_prompt = f"""
    Repository: {repo_name}
    
    Analyze the following activity log:
    {context_text}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # 3. Invoke LLM
    response = llm.invoke(messages)
    
    # 4. Update State (Append the analysis to the logs)
    return {"analysis_logs": [response.content]}


def generate_report_node(state: GraphState):
    """
    Reporter Node: Synthesizes the analysis into the final structured object.
    """
    print("📝 Agent: Generating structured report...")
    
    developer_name = state["developer_name"]
    analysis_text = "\n".join(state["analysis_logs"])
    
    # In a real scenario, we would use 'with_structured_output' (Pydantic) 
    # but for this MVP step, we will mock the object creation based on the text 
    # to keep it simple, or use a second LLM call to structure it.
    
    # Let's do a basic mapping for the MVP
    report = DeveloperReport(
        developer_name=developer_name,
        period_start=datetime.now(), # Placeholder
        period_end=datetime.now(),
        tasks_completed=0, # We would calculate this from Trello adapter later
        prs_merged=len([a for a in state["activities"] if "MERGE" in a.type.value]),
        quality_score=8, # Placeholder: In next steps we ask LLM to output this number
        feedback_summary=analysis_text, # The raw analysis from the previous node
        detected_skills=["Python", "Git", "Hexagonal Arch"] # Placeholder
    )
    
    return {"final_report": report}