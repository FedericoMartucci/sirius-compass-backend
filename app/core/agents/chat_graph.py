# app/core/agents/chat_graph.py
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from app.core.agents.state import ChatState
from app.core.llm_factory import get_llm
# Import both tools.
from app.tools.db_reader import (
    get_developer_activity,
    get_developer_code_activity_by_project,
    get_developer_recent_work,
    get_latest_audit_report,
    get_project_developers,
    get_repository_activity,
)
from app.tools.sync_manager import refresh_repository_data

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
        get_project_developers,
        refresh_repository_data,
        get_repository_activity,
    ]
    
    llm = get_llm(temperature=0, streaming=True).bind_tools(tools)

    # 2. Agent Node
    def agent_node(state: ChatState):
        system_msg = SystemMessage(content="""
        You are Sirius Compass.
        
        CAPABILITIES:
        1. General Audit: Use 'get_latest_audit_report' for scores/summaries.
        2. Developer Commits: Use 'get_developer_activity' for a repo-scoped view (requires developer name).
        3. Project Commits: Use 'get_developer_code_activity_by_project' for project-scoped commits across repos.
        4. Commits + Tickets: Use 'get_developer_recent_work' when the user asks for commits AND related tickets.
        5. Developers List: Use 'get_project_developers' when asked who is working on a project.
        6. Refresh Data: Use 'refresh_repository_data' if the user asks for "latest", "newest", "just now", or implies the data is old.
        7. Global Activity: Use 'get_repository_activity' to see the latest commits from ANYONE in a repo. Use this if no specific developer is mentioned.

        IMPORTANT:
        - If the user explicitly asks for the LATEST info or complains about stale data, CALL 'refresh_repository_data' FIRST, then call 'get_repository_activity' or other reader tools.
        - If the user asks "which tickets" (or similar), do NOT guess from branch names. Use 'get_developer_recent_work'.
        
        Try to answer the user's question directly using the data from the tools.

        Always respond in Spanish.
        """)
        
        return {"messages": [llm.invoke([system_msg] + state["messages"])]}

    # Graph wiring.
    workflow = StateGraph(ChatState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()
