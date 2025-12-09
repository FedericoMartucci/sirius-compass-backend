from abc import ABC, abstractmethod
from typing import List
from app.core.models.domain import UnifiedActivity

class CodeProvider(ABC):
    """
    Port (Interface) for any code provider.
    The Core logic will depend on this abstraction, never on the concrete GitHub implementation.
    """
    
    @abstractmethod
    def fetch_recent_activity(self, repo_name: str, days: int = 7) -> List[UnifiedActivity]:
        """
        Fetches recent activity (commits, PRs) from a repository.
        
        Args:
            repo_name (str): The repository identifier (e.g., "owner/repo").
            days (int): Time window in days to look back.
            
        Returns:
            List[UnifiedActivity]: A list of normalized activity objects.
        """
        pass