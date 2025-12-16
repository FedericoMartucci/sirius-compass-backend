import os
import requests
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
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
    state_type: Optional[str] = None
    createdAt: str
    updatedAt: Optional[str] = None
    completedAt: Optional[str] = None
    url: str
    assignee: Optional[str] = None
    cycle_name: Optional[str] = None
    cycle_number: Optional[int] = None
    cycle_startsAt: Optional[str] = None
    cycle_endsAt: Optional[str] = None

class LinearAdapter:
    """
    Adapter for Linear's GraphQL API.
    """
    BASE_URL = "https://api.linear.app/graphql"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("LINEAR_API_KEY")
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
        issues, _, _ = self.fetch_issues_page(
            limit=limit,
            team_key=team_key,
            updated_since=updated_since,
            after=None,
        )
        return issues

    def fetch_issues_page(
        self,
        *,
        limit: int = 50,
        team_key: Optional[str] = None,
        updated_since: Optional[datetime] = None,
        after: Optional[str] = None,
    ) -> Tuple[List[LinearIssue], Optional[str], bool]:
        if not self._enabled:
            return [], None, False

        variable_defs = ["$limit: Int!"]
        variables: Dict[str, Any] = {"limit": limit}

        if after:
            variable_defs.append("$after: String!")
            variables["after"] = after

        # NOTE: Linear's `WorkflowStateType` is an enum; enum values must NOT be quoted.
        # Actually, the error `String cannot represent a non string value: canceled` suggests it WANTS a string.
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
          issues(first: $limit{", after: $after" if after else ""}, orderBy: updatedAt, filter: {{
              {filter_block}
          }}) {{
            nodes {{
              id
              identifier
              title
              description
              estimate
              state {{ name type }}
              createdAt
              updatedAt
              completedAt
              url
              cycle {{ name number startsAt endsAt }}
              assignee {{ name }}
            }}
            pageInfo {{
              hasNextPage
              endCursor
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
            if not response.ok:
                body = response.text
                if len(body) > 1500:
                    body = body[:1500] + "... (truncated)"
                logger.error(f"❌ Linear HTTP Error {response.status_code}: {body}")
                response.raise_for_status()

            payload = response.json() if response.content else {}
            if payload.get("errors"):
                logger.error(f"❌ Linear GraphQL Error: {payload.get('errors')}")
                return [], None, False

            issues_data = payload.get("data", {}).get("issues", {}) or {}
            data = issues_data.get("nodes", []) or []
            page_info = issues_data.get("pageInfo", {}) or {}
            has_next = bool(page_info.get("hasNextPage"))
            end_cursor = page_info.get("endCursor")
            
            issues = []
            for i in data:
                cycle = i.get("cycle") or {}
                issues.append(
                    LinearIssue(
                        id=i["id"],
                        identifier=i["identifier"],
                        title=i["title"],
                        description=i.get("description"),
                        estimate=i.get("estimate") or 0,
                        state=(i.get("state") or {}).get("name") or "unknown",
                        state_type=(i.get("state") or {}).get("type"),
                        createdAt=i["createdAt"],
                        updatedAt=i.get("updatedAt"),
                        completedAt=i.get("completedAt"),
                        url=i["url"],
                        cycle_name=cycle.get("name"),
                        cycle_number=cycle.get("number"),
                        cycle_startsAt=cycle.get("startsAt"),
                        cycle_endsAt=cycle.get("endsAt"),
                        assignee=(i.get("assignee") or {}).get("name"),
                    )
                )

            return issues, end_cursor, has_next
        except Exception as e:
            logger.error(f"❌ Linear Sync Error: {e}")
            raise e
