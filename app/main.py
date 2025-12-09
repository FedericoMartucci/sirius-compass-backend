"""
Entry point for the Sirius Compass backend.

The bootstrap function wires adapters and agents. Replace placeholders with
actual implementations as each port/adapter is developed.
"""

from typing import Any


def bootstrap() -> dict[str, Any]:
    """Return a simple runtime context; extend with real dependencies."""
    return {
        "adapters": {
            "github": None,
            "trello": None,
        },
        "agents": {
            "chat": None,
            "reporting": None,
        },
    }


if __name__ == "__main__":
    context = bootstrap()
    print("Sirius Compass backend initialized:", context)
