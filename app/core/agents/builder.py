from langgraph.graph import StateGraph, END
from app.core.agents.state import GraphState
from app.core.agents.nodes import analyze_activities_node, generate_report_node

def build_sirius_graph():
    """
    Constructs the LangGraph workflow.
    Flow: Start -> Analyze -> Report -> End
    """
    # 1. Initialize the Graph with our typed State
    workflow = StateGraph(GraphState)

    # 2. Add Nodes
    workflow.add_node("analyst", analyze_activities_node)
    workflow.add_node("reporter", generate_report_node)

    # 3. Define Edges (The flow)
    workflow.set_entry_point("analyst") # Start here
    workflow.add_edge("analyst", "reporter") # Go to reporter
    workflow.add_edge("reporter", END) # Finish

    # 4. Compile (Freeze the graph)
    app = workflow.compile()
    return app