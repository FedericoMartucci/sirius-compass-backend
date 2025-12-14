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
    get_latest_audit_report,
)

def build_chat_graph():
    """
    Interactive Chatbot with Granular DB Access.
    """
    # 1. Setup tools (includes get_developer_activity).
    tools = [get_latest_audit_report, get_developer_activity, get_developer_code_activity_by_project]
    
    llm = get_llm(temperature=0, streaming=True).bind_tools(tools)

    # 2. Agent Node
    def agent_node(state: ChatState):
        system_msg = SystemMessage(content="""
        You are Sirius Compass.
        
        CAPABILITIES:
        1. General Audit: Use 'get_latest_audit_report' for scores/summaries.
        2. Specific Investigation: Use 'get_developer_activity' when asked about specific people/commits.
        3. Project Scope Investigation: Use 'get_developer_code_activity_by_project' when a project has multiple repos.
        
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
