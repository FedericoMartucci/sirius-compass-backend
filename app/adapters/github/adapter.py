import os
from typing import List
from datetime import datetime, timedelta, timezone
from github import Github, GithubException
from app.ports.code_provider import CodeProvider
from app.core.models.domain import UnifiedActivity, ActivityType

class GitHubAdapter(CodeProvider):
    """
    Concrete implementation of CodeProvider for GitHub.
    Uses PyGithub to fetch data and maps it to the Unified Sirius Domain Model.
    """

    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("Missing GITHUB_TOKEN in environment variables.")
        self.client = Github(token)

    def _extract_diff(self, files_object, limit_files: int = 10) -> str:
        """
        It extracts changed code.
        """
        diff_summary = []
        count = 0
        for file in files_object:
            if count >= limit_files:
                diff_summary.append("... (more files truncated)")
                break
            
            if file.filename.endswith(('.lock', '.png', '.jpg', 'package-lock.json')):
                continue

            patch = file.patch if file.patch else "[Binary or Large File]"
            
            if len(patch) > 2000:
                patch = patch[:2000] + "\n... (truncated)"

            diff_summary.append(f"File: {file.filename}\nChanges:\n{patch}")
            count += 1
            
        return "\n\n".join(diff_summary)

    def fetch_recent_activity(self, repo_name: str, days: int = 7) -> List[UnifiedActivity]:
        print(f"Connecting to GitHub Repo: {repo_name}...")
        activities = []
        try:
            repo = self.client.get_repo(repo_name)
            since_date = datetime.now(timezone.utc) - timedelta(days=days)

            # Fetch Commits
            commits = repo.get_commits(since=since_date)
            for commit in commits:
                diff_content = self._extract_diff(commit.files)
                activity = UnifiedActivity(
                    source_id=commit.sha,
                    author=commit.commit.author.name or "Unknown",
                    type=ActivityType.COMMIT,
                    content=f"Message: {commit.commit.message}\n\nCode Diff:\n{diff_content}",
                    timestamp=commit.commit.author.date.replace(tzinfo=timezone.utc),
                    url=commit.html_url,
                    files_changed=[f.filename for f in commit.files[:10]]
                )
                activities.append(activity)

            # Fetch Pull Requests
            # We fetch 'all' and filter by date locally because the API filter is limited
            prs = repo.get_pulls(state='all', sort='updated', direction='desc')
            for pr in prs:
                # Ensure PR date is timezone-aware for comparison
                pr_updated_at = pr.updated_at.replace(tzinfo=timezone.utc)
                
                if pr_updated_at < since_date:
                    break # Optimization: Stop if we reached older PRs
                
                # Determine type: Merged, Open, or Closed (we treat closed as PR_OPEN)
                type_pr = ActivityType.PR_MERGE if pr.merged else ActivityType.PR_OPEN
                
                activity = UnifiedActivity(
                    source_id=str(pr.number),
                    author=pr.user.login,
                    type=type_pr,
                    content=f"Title: {pr.title}\nBody: {pr.body}\n\nCode Diff:\n{diff_content}",
                    timestamp=pr.created_at.replace(tzinfo=timezone.utc),
                    url=pr.html_url,
                    additions=pr.additions,
                    deletions=pr.deletions,
                    files_changed=[f.filename for f in pr.get_files()[:10]] 
                )
                activities.append(activity)

            print(f"Found {len(activities)} activities in {repo_name}")
            return activities

        except GithubException as e:
            print(f"GitHub API Error: {e.data.get('message', e)}")
            return []
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return []