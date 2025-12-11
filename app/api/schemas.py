# app/api/schemas.py
from pydantic import BaseModel
from typing import Optional, Dict, Any

class AnalyzeRequest(BaseModel):
    repo_url: str
    developer_name: str = "Unknown"
    lookback_days: int = 7

class AnalyzeResponse(BaseModel):
    status: str
    report: Dict[str, Any]
    metadata: Dict[str, Any]