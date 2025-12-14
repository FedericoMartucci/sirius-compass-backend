from typing import Optional
from langchain_core.tools import tool
from sqlmodel import Session, select, col
from app.core.database.models import (
    Activity,
    AnalysisReport,
    Project,
    ProjectRepository,
    Repository,
)
from app.core.database.session import engine

@tool
def get_latest_audit_report(repo_name: str):
    """
    Fetches the high-level audit summary and score for a repository.
    Use this for general questions like "How is the project doing?" or "What is the quality score?".
    """
    with Session(engine) as session:
        repo = session.exec(select(Repository).where(Repository.name.contains(repo_name))).first()
        if not repo: return "Repo not found."

        report = session.exec(
            select(AnalysisReport)
            .where(AnalysisReport.repository_id == repo.id)
            .order_by(AnalysisReport.id.desc())
        ).first()
        
        if not report: return "No audit found."
        return f"DEV: {report.developer_name}\nSCORE: {report.quality_score}\nSUMMARY: {report.feedback_summary}"

@tool
def get_developer_activity(repo_name: str, developer_name: str, limit: int = 7):
    """
    Fetches raw activity (commits, tickets) for a specific developer in a repository.
    Use this when the user asks "What did Fede do?" or "Show me the last commits of X".
    """
    with Session(engine) as session:
        # 1. Find Repo
        repo = session.exec(select(Repository).where(Repository.name.contains(repo_name))).first()
        if not repo: return "Repository not found."
        
        # 2. Find Activities (Case insensitive author search)
        statement = (
            select(Activity)
            .where(Activity.repository_id == repo.id)
            .where(col(Activity.author).ilike(f"%{developer_name}%"))
            .order_by(Activity.timestamp.desc())
            .limit(limit)
        )
        activities = session.exec(statement).all()
        
        if not activities:
            return f"No activities found for developer '{developer_name}' in '{repo_name}'."
            
        # 3. Format for LLM
        output = f"Last {len(activities)} activities for {developer_name}:\n"
        for act in activities:
            output += f"- [{act.type}] {act.timestamp.date()}: {act.title} (Source: {act.source_platform})\n"
            
        return output


@tool
def get_developer_code_activity_by_project(project_name: str, developer_name: str, limit: int = 7):
    """
    Fetches GitHub activity across all repositories linked to an internal Project.

    Use this when the user asks about a developer and the scope is a "project"
    that contains multiple repositories.
    """
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name.contains(project_name))).first()
        if not project:
            return "Project not found."

        repo_ids = session.exec(
            select(ProjectRepository.repository_id).where(ProjectRepository.project_id == project.id)
        ).all()
        repo_ids = [rid for rid in repo_ids if rid is not None]
        if not repo_ids:
            return "No repositories linked to this project."

        statement = (
            select(Activity)
            .where(Activity.repository_id.in_(repo_ids))
            .where(col(Activity.author).ilike(f"%{developer_name}%"))
            .order_by(Activity.timestamp.desc())
            .limit(limit)
        )
        activities = session.exec(statement).all()

        if not activities:
            return f"No activities found for developer '{developer_name}' in project '{project_name}'."

        repos = session.exec(select(Repository).where(Repository.id.in_(repo_ids))).all()
        repo_name_by_id = {r.id: r.name for r in repos}

        output = f"Last {len(activities)} activities for {developer_name} in project {project.name}:\n"
        for act in activities:
            repo_name = repo_name_by_id.get(act.repository_id, str(act.repository_id))
            output += f"- [{act.type}] {act.timestamp.date()}: {act.title} (Repo: {repo_name})\n"

        return output
