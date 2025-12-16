from sqlmodel import Session, select
from app.api.server import engine
from app.core.database.models import Project, Repository

def inspect_projects():
    with Session(engine) as session:
        projects = session.exec(select(Project)).all()
        print(f"Found {len(projects)} projects:")
        for p in projects:
            print(f"ID: {p.id}, Name: '{p.name}'")

        repos = session.exec(select(Repository)).all()
        print(f"\nFound {len(repos)} repositories:")
        for r in repos:
            print(f"ID: {r.id}, Name: '{r.name}', URL: '{r.url}'")

if __name__ == "__main__":
    inspect_projects()
