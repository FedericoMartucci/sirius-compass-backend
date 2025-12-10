from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.agents.state import GraphState
from app.core.llm_factory import get_llm
from app.core.models.domain import DeveloperReport
from app.core.rag.vector_store import search_similar_rules

def analyze_activities_node(state: GraphState):
    """
    Analyst Node: Takes raw activities (with code diffs), retrieves Sirius Standards via RAG,
    and asks the LLM to audit the code against those specific rules.
    """
    print("🧠 Agent: Analyzing code patterns and diffs (Sirius Context Mode)...")
    
    activities = state["activities"]
    repo_name = state["repo_name"]
    
    # Temperature 0 for strict objectivity
    llm = get_llm(temperature=0) 
    
    # 1. RAG RETRIEVAL (Fetch Sirius Standards)
    # We query for general quality, security, and git standards to have the full context
    print("📚 Retrieving Sirius Engineering Standards from Pinecone...")
    retrieved_rules = search_similar_rules("security credentials git commit style testing rules errors", k=8)

    # 2. PREPARE CONTEXT (Raw Activity Data)
    context_text = ""
    # We limit to 15 activities to fit context window while giving enough history
    for act in activities[:15]:
        context_text += f"""
        ---
        [Activity Type]: {act.type.value}
        [Date]: {act.timestamp}
        [Content/Diff Summary]: 
        {act.content[:2000]} 
        ---
        """

    # 3. DEFINE PROMPT (Sirius Tech Lead Persona)
    system_prompt = f"""
    You are Sirius Compass, a Senior Tech Lead at Sirius Software.
    Your role is to audit code with a PRAGMATIC but SECURE mindset, strictly following internal culture.
    
    INTERNAL SIRIUS STANDARDS (You must enforce these):
    {retrieved_rules}
    
    AUDIT GUIDELINES:
    1. SECURITY IS NON-NEGOTIABLE[cite: 100]:
       - Flag IMMEDIATELY any committed .env, API Key, or secret. This is a critical failure.
       - Flag any logging of PII (passwords, emails).
       
    2. LANGUAGE & CONSISTENCY[cite: 4]:
       - Detect the dominant language of the repo (English or Spanish).
       - Flag "Spanglish" (mixing languages in variables/files) as a Code Smell.
       - If the repo is in Spanish (LATAM client), Spanish variable names are ACCEPTABLE. Do not flag them unless mixed with English.
       
    3. GIT & WORKFLOW[cite: 27]:
       - "Mega-Commits" (many files changed, vague message) are a yellow card.
       - Conventional Commits (feat:, fix:) are desired.
       
    4. TESTING & PRAGMATISM[cite: 46]:
       - If a commit message says "HOTFIX" or "URGENT", accept lack of tests but add a warning to add them later.
       - For normal features, complain if there are NO tests or if error handling is swallowed (empty catch blocks).
       
    5. PROFILE[cite: 128]:
       - Value "Self-documenting code" over excessive comments.
       - We prefer "Move fast, but don't break things". Do not be a purist about minor style issues if the code is solid and secure.

    OUTPUT FORMAT (In Spanish):
    - **Score de Calidad (1-10)**: Be strict on security, flexible on style.
    - **Alerta de Seguridad**: (YES/NO) - Details if YES.
    - **Análisis de Consistencia**: (Check for Spanglish).
    - **Puntos Fuertes**: (e.g. Atomic commits, good error handling).
    - **A mejorar**: (Specific files/lines).
    - **Veredicto**: (Ready to Merge / Needs Work / Blocked).
    """

    user_prompt = f"""
    Repository: {repo_name}
    
    Raw Activity Log:
    {context_text}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # 4. Invoke LLM
    response = llm.invoke(messages)
    
    # 5. Update State
    return {"analysis_logs": [response.content]}

def generate_report_node(state: GraphState):
    """
    Reporter Node: Synthesizes the analysis into the final structured object.
    (Mocking data for MVP flow until we implement Structured Output)
    """
    print("📝 Agent: Generating structured report...")
    
    developer_name = state["developer_name"]
    analysis_text = "\n".join(state["analysis_logs"])
    
    # Placeholder report logic for MVP
    report = DeveloperReport(
        developer_name=developer_name,
        period_start=datetime.now(),
        period_end=datetime.now(),
        tasks_completed=0,
        prs_merged=len([a for a in state["activities"] if "MERGE" in a.type.value]),
        quality_score=0, # The score is currently embedded in the text analysis
        feedback_summary=analysis_text,
        detected_skills=["Python", "LangGraph", "RAG"] # Placeholder
    )
    
    return {"final_report": report}