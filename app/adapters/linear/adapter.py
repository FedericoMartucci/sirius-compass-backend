import os
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.core.logger import get_logger

logger = get_logger(__name__)

class LinearIssue(BaseModel):
    id: str
    identifier: str
    title: str
    description: Optional[str] = None
    estimate: int = 0
    state: str
    createdAt: str
    completedAt: Optional[str] = None
    url: str

class LinearAdapter:
    """
    Adapter for Linear's GraphQL API.
    """
    BASE_URL = "https://api.linear.app/graphql"

    def __init__(self):
        self.api_key = os.getenv("LINEAR_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ LINEAR_API_KEY not found. Velocity metrics will be skipped.")
            self._enabled = False
        else:
            self._enabled = True
            self._headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json"
            }

    def fetch_recent_issues(self, limit: int = 50) -> List[LinearIssue]:
        if not self._enabled: return []

        query = """
        query Issues($limit: Int!) {
          issues(first: $limit, orderBy: updatedAt, filter: { state: { type: { neq: "canceled" } } }) {
            nodes {
              id
              identifier
              title
              description
              estimate
              state { name }
              createdAt
              completedAt
              url
            }
          }
        }
        """
        try:
            response = requests.post(
                self.BASE_URL, 
                json={"query": query, "variables": {"limit": limit}}, 
                headers=self._headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json().get("data", {}).get("issues", {}).get("nodes", [])
            
            return [
                LinearIssue(
                    id=i["id"], identifier=i["identifier"], title=i["title"],
                    description=i.get("description"), estimate=i.get("estimate") or 0,
                    state=i["state"]["name"], createdAt=i["createdAt"],
                    completedAt=i.get("completedAt"), url=i["url"]
                ) for i in data
            ]
        except Exception as e:
            logger.error(f"❌ Linear Sync Error: {e}")
            return []