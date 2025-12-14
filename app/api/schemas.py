from pydantic import BaseModel
from typing import Dict, Any, Optional

class AnalyzeRequest(BaseModel):
    repo_url: str
    developer_name: str = "Unknown"
    lookback_days: int = 720 
    project_name: Optional[str] = None
    linear_team_key: Optional[str] = None

class AnalyzeResponse(BaseModel):
    status: str
    report: Dict[str, Any]
    metadata: Dict[str, Any]
    message: Optional[str] = None
    report_summary: Optional[str] = None

class ChatRequest(BaseModel):
    thread_id: str
    message: str
    repo_name: str
    user_id: Optional[str] = None
    project_name: Optional[str] = None
