import operator
from typing import Any, Annotated, Dict, List, Optional, TypedDict
from langgraph.graph.message import add_messages
from app.core.models.domain import UnifiedActivity, DeveloperReport

class GraphState(TypedDict):
    """
    State for the Analyst Graph (Batch Process).
    """
    repo_name: str
    project_name: str
    developer_name: str
    lookback_days: int
    linear_team_key: Optional[str]
    user_id: Optional[str]
    
    # 'operator.add' allows parallel nodes to append to this list instead of overwriting
    activities: Annotated[List[UnifiedActivity], operator.add]
    
    analysis_logs: Annotated[List[str], operator.add]
    structured_analysis: Optional[Dict[str, Any]]
    final_report: Optional[DeveloperReport]

class ChatState(TypedDict, total=False):
    """
    State for the Chat Graph (Conversational Process).
    Stores message history.
    """
    messages: Annotated[list, add_messages]
    meta: Dict[str, Any]
    skip_agent: bool
