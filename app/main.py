import os
from dotenv import load_dotenv
from app.adapters.github.adapter import GitHubAdapter
from app.core.agents.builder import build_sirius_graph

load_dotenv()

def run_mvp_flow():
    """
    Runs the full flow: GitHub -> Adapter -> LangGraph -> Report
    """
    print("🚀 Sirius Compass: Starting MVP Flow...")
    
    # 1. Fetch Data (Infrastructure Layer)
    github_adapter = GitHubAdapter()
    repo_name = "FedericoMartucci/sirius-compass-backend"
    print(f"📥 Fetching data from {repo_name}...")
    
    activities = github_adapter.fetch_recent_activity(repo_name, days=3)
    
    if not activities:
        print("⚠️ No activities found. Exiting.")
        return

    # 2. Initialize Agent State
    initial_state = {
        "repo_name": repo_name,
        "developer_name": "Unknown Dev", # In real app, we filter by author
        "activities": activities,
        "analysis_logs": [],
        "final_report": None
    }

    # 3. Run the Graph (Core Layer)
    print("🤖 Invoking AI Agent...")
    app = build_sirius_graph()
    result = app.invoke(initial_state)

    # 4. Show Output
    report = result["final_report"]
    print("\n" + "="*50)
    print(f"📊 REPORT FOR: {repo_name}")
    print("="*50)
    print(f"📝 Feedback Summary:\n{report.feedback_summary}")
    print("="*50)
    print(f"🔢 PRs Merged: {report.prs_merged}")
    print("="*50)

if __name__ == "__main__":
    run_mvp_flow()