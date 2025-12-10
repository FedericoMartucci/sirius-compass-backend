from dotenv import load_dotenv
from app.core.rag.vector_store import add_documents

load_dotenv()

def seed_engineering_standards():
    """
    Populates Pinecone with Sirius Engineering Standards.
    Run this script ONCE to initialize the knowledge base.
    """
    print("🌱 Seeding Database with Standards...")
    
    standards = [
        "Language Consistency: All code, comments, and documentation must be written in English. No Spanish allowed in the codebase.",
        "Repository Hygiene: Do not commit configuration files (.env), pycache, or large binary files.",
        "Testing: All new features must include unit tests using pytest. Coverage should not decrease.",
        "Atomic Commits: Commits should focus on a single logical change. Avoid 'Mega-Commits'.",
        "Error Handling: Never use bare 'except:' clauses. Always catch specific exceptions.",
        "Clean Code: Functions should be small and do one thing. Use descriptive variable names (no 'x', 'temp')."
    ]
    
    metadatas = [{"category": "general"} for _ in standards]
    
    try:
        add_documents(texts=standards, metadatas=metadatas)
        print("Database seeded successfully!")
    except Exception as e:
        print(f"Error seeding database: {e}")

if __name__ == "__main__":
    seed_engineering_standards()