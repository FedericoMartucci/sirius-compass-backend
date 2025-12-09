from dotenv import load_dotenv
from app.adapters.github.adapter import GitHubAdapter

load_dotenv()

def test_github_integration():
    try:
        adapter = GitHubAdapter()
        
        # CAMBIA ESTO por un repo real tuyo o público (ej. "langchain-ai/langchain")
        repo_to_test = "federicomartucci/sirius-compass-backend" 
        
        print(f"🧪 Testing GitHub Adapter on: {repo_to_test}")
        activities = adapter.fetch_recent_activity(repo_name=repo_to_test, days=2)
        
        if not activities:
            print("⚠️ No activity found (or error occurred). Check repository name and dates.")
            return

        print(f"\nLast 3 activities found:")
        for act in activities[:3]:
            print(f"  - [{act.type.value}] {act.author}: {act.content[:40]}... ({act.timestamp})")
            print(f"{activities}")
            
    except Exception as e:
        print(f"❌ Critical Test Error: {e}")

if __name__ == "__main__":
    test_github_integration()