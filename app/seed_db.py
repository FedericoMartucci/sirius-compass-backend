from dotenv import load_dotenv
from app.core.rag.vector_store import add_documents

load_dotenv()

def seed_sirius_standards():
    """
    Populates Pinecone with granular chunks of Sirius Software's DNA.
    Derived directly from 'Cultura y Estándares de Desarrollo en Sirius Software.pdf'.
    """
    print("🌱 Seeding Database with Granular Sirius Standards...")
    
    # Granular chunks based on Sirius internal PDF
    standards_data = [
        # --- 1. LANGUAGE & CONSISTENCY ---
        {
            "text": "LANGUAGE POLICY: The language is defined by the client context (LATAM=Spanish, International=English). Crucially, consistency is mandatory: mixing languages (Spanglish) within the same repo for variables or files is a bad practice[cite: 4, 6, 9]. The agent must detect the project language and enforce it.",
            "meta": {"category": "style", "severity": "high"}
        },
        
        # --- 2. GIT WORKFLOW ---
        {
            "text": "COMMIT MESSAGES: Use descriptive messages, preferably following Conventional Commits (feat:, fix:). Vague messages are discouraged. New members are directed to the Conventional Commits spec[cite: 14, 18].",
            "meta": {"category": "git", "severity": "medium"}
        },
        {
            "text": "BRANCHING STRATEGY: There is no single imposed flow. Some projects use GitFlow (develop/release), others Trunk-Based (main). The agent must detect the pattern (e.g., existence of 'develop' branch) and adapt[cite: 20, 25].",
            "meta": {"category": "git", "severity": "medium"}
        },
        {
            "text": "ATOMIC COMMITS: Granularity is valued. A 'Mega-Commit' changing 50 files is a 'yellow card'. Ideally, one commit per specific feature or fix[cite: 27, 28]. Avoid mixing massive changes.",
            "meta": {"category": "git", "severity": "high"}
        },

        # --- 3. TESTING ---
        {
            "text": "TESTING REQUIREMENTS: Automated tests (Unit/Integration) are highly valued and expected (~80% coverage goal), even if not contractually required. A new module without tests raises red flags in Code Review[cite: 34, 40].",
            "meta": {"category": "testing", "severity": "high"}
        },
        {
            "text": "TEST QUALITY: Presence of tests is not enough; they must be readable and have assertions. Tests covering the 'Happy Path' and critical cases are the priority over exact coverage numbers[cite: 38, 41].",
            "meta": {"category": "testing", "severity": "medium"}
        },
        {
            "text": "HOTFIX EXCEPTION: Merging a hotfix without tests is tolerated ONLY during urgent production fires[cite: 46, 47]. However, it is expected that tests are added immediately after the emergency is resolved.",
            "meta": {"category": "testing", "severity": "critical"}
        },

        # --- 4. TECH STACK ---
        {
            "text": "FRONTEND STACK: React is the standard. Next.js is also used in production. Angular appears occasionally for legacy/specific clients. Assume React/TypeScript unless stated otherwise[cite: 55, 60].",
            "meta": {"category": "stack", "severity": "low"}
        },
        {
            "text": "BACKEND STACK: Heterogeneous. Node.js (Express/NestJS) with TypeScript is common for new projects. Python (Django/FastAPI) and Java (Spring Boot) or C# (.NET) are used based on client needs[cite: 62, 64, 65]. No proprietary internal frameworks are imposed.",
            "meta": {"category": "stack", "severity": "low"}
        },
        {
            "text": "MOBILE STACK: React Native (specifically Expo) is the preferred choice to leverage web know-how[cite: 69, 70]. Native (Swift/Kotlin) is rare but possible.",
            "meta": {"category": "stack", "severity": "low"}
        },

        # --- 5. ERROR HANDLING ---
        {
            "text": "EXCEPTION HANDLING: Never silence errors. Empty 'try/except' or 'catch(e) {}' blocks are forbidden. Errors must be explicitly handled or logged. Swallowing exceptions is strictly criticized in Code Reviews[cite: 85, 89].",
            "meta": {"category": "quality", "severity": "critical"}
        },
        {
            "text": "LOGGING TOOLS: Sentry is the standard for error capturing. Use structured logging (logger.error) instead of console.log/print so errors reach Sentry/Datadog[cite: 92, 94].",
            "meta": {"category": "quality", "severity": "high"}
        },

        # --- 6. SECURITY (GUARDRAILS) ---
        {
            "text": "NO SECRETS IN REPO: Committing .env files, API keys, or secrets is the gravest error. Keys must be managed via environment variables/vaults. Any hardcoded secret (e.g., AWS Key) triggers an immediate Red Alert[cite: 100, 107].",
            "meta": {"category": "security", "severity": "critical"}
        },
        {
            "text": "DATA PRIVACY: Do not log PII (emails, passwords, DNI) or dump full user objects to the console/logs. This exposes the company to data leaks. Sensitive data must be sanitized[cite: 112, 118].",
            "meta": {"category": "security", "severity": "critical"}
        },

        # --- 7. CULTURE & PROFILE ---
        {
            "text": "SPEED VS PERFECTION: 'Move fast, but don't break things (too much)'. Delivering working solutions is priority over gold-plating, but code must be solid[cite: 128]. Avoid perfectionism that causes inaction, but also avoid reckless haste.",
            "meta": {"category": "culture", "severity": "medium"}
        },
        {
            "text": "DOCUMENTATION STYLE: Prefer self-documenting code (clear names) over excessive comments. Redundant comments explaining the obvious are a red flag[cite: 133, 137]. Comments should explain 'Why', not 'What'.",
            "meta": {"category": "style", "severity": "medium"}
        }
    ]
    
    texts = [r["text"] for r in standards_data]
    metadatas = [r["meta"] for r in standards_data]
    
    try:
        add_documents(texts=texts, metadatas=metadatas)
        print(f"🚀 Sirius DNA seeded successfully! ({len(texts)} granular chunks)")
    except Exception as e:
        print(f"❌ Error seeding database: {e}")

if __name__ == "__main__":
    seed_sirius_standards()