from app.main import bootstrap


def test_bootstrap_returns_context():
    context = bootstrap()

    assert "adapters" in context
    assert "agents" in context
