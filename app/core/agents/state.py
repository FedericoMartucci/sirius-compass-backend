import operator
from typing import List, Annotated, Optional, TypedDict
from app.core.models.domain import UnifiedActivity, DeveloperReport

class GraphState(TypedDict):
    """
    Represents the shared memory of the agent workflow.
    LangGraph passes this dictionary between nodes.
    """
    repo_name: str
    developer_name: str
    activities: List[UnifiedActivity]  # The raw data from GitHub/Trello

    # Internal Memory
    analysis_logs: Annotated[List[str], operator.add]
    
    # Outputs
    final_report: Optional[DeveloperReport]