import os
import requests
from datetime import datetime
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
    updatedAt: Optional[str] = None
    completedAt: Optional[str] = None
    url: str
    assignee: Optional[str] = None

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

    def fetch_recent_issues(
        self,
        limit: int = 50,
        team_key: Optional[str] = None,
        updated_since: Optional[datetime] = None,
    ) -> List[LinearIssue]:
        if not self._enabled: return []

        variable_defs = ["$limit: Int!"]
        variables: Dict[str, Any] = {"limit": limit}

        filter_lines = ['state: { type: { neq: "canceled" } }']
        if team_key:
            variable_defs.append("$teamKey: String!")
            variables["teamKey"] = team_key
            filter_lines.append("team: { key: { eq: $teamKey } }")

        if updated_since:
            variable_defs.append("$updatedSince: DateTime!")
            variables["updatedSince"] = updated_since.isoformat()
            filter_lines.append("updatedAt: { gte: $updatedSince }")

        filter_block = "\n              ".join(filter_lines)
        query = f"""
        query Issues({", ".join(variable_defs)}) {{
          issues(first: $limit, orderBy: updatedAt, filter: {{
              {filter_block}
          }}) {{
            nodes {{
              id
              identifier
              title
              description
              estimate
              state {{ name }}
              createdAt
              updatedAt
              completedAt
              url
              assignee {{ name }}
            }}
          }}
        }}
        """
        try:
            response = requests.post(
                self.BASE_URL, 
                json={"query": query, "variables": variables},
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
                    updatedAt=i.get("updatedAt"),
                    completedAt=i.get("completedAt"),
                    url=i["url"],
                    assignee=(i.get("assignee") or {}).get("name"),
                ) for i in data
            ]
        except Exception as e:
            logger.error(f"❌ Linear Sync Error: {e}")
            return []
