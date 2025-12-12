import os
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.core.logger import get_logger

logger = get_logger(__name__)

class LinearIssue(BaseModel):
    id: str
    identifier: str # e.g., LIN-123
    title: str
    description: Optional[str] = None
    estimate: int = 0 # Story Points
    state: str
    createdAt: str
    completedAt: Optional[str] = None
    url: str

class LinearAdapter:
    """
    Adapter for Linear's GraphQL API.
    Used to ingest tickets and estimates for Velocity tracking.
    """
    BASE_URL = "https://api.linear.app/graphql"

    def __init__(self):
        self.api_key = os.getenv("LINEAR_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ LINEAR_API_KEY not found. Velocity metrics will be disabled.")
            self._enabled = False
        else:
            self._enabled = True
            self._headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json"
            }

    def _execute_query(self, query: str, variables: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Executes a generic GraphQL query.
        """
        if not self._enabled:
            return {}

        try:
            response = requests.post(
                self.BASE_URL, 
                json={"query": query, "variables": variables}, 
                headers=self._headers,
                timeout=10
            )
            response.raise_for_status()
            
            payload = response.json()
            if "errors" in payload:
                logger.error(f"❌ Linear GraphQL Errors: {payload['errors']}")
                return {}
                
            return payload.get("data", {})

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Linear Connection Error: {e}")
            return {}

    def fetch_recent_issues(self, team_key: Optional[str] = None, limit: int = 50) -> List[LinearIssue]:
        """
        Fetches recent issues. If team_key is provided, filters by team.
        """
        # GraphQL Query to fetch issues
        # We assume 'first: 50' for MVP simplicity
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
              team { key } 
            }
          }
        }
        """
        variables = {"limit": limit}
        data = self._execute_query(query, variables)
        
        issues_data = data.get("issues", {}).get("nodes", [])
        
        parsed_issues = []
        for issue in issues_data:
            # Filter by team_key manually if API filter is complex
            if team_key and issue.get("team", {}).get("key") != team_key:
                continue

            try:
                parsed_issues.append(LinearIssue(
                    id=issue["id"],
                    identifier=issue["identifier"],
                    title=issue["title"],
                    description=issue.get("description"),
                    estimate=issue.get("estimate") or 0, # Handle None as 0 points
                    state=issue["state"]["name"],
                    createdAt=issue["createdAt"],
                    completedAt=issue["completedAt"],
                    url=issue["url"]
                ))
            except Exception as e:
                logger.warning(f"Skipping malformed issue {issue.get('identifier', 'unknown')}: {e}")

        logger.info(f"✅ Fetched {len(parsed_issues)} issues from Linear")
        return parsed_issues