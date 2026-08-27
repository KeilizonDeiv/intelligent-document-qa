import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings


def parse_sse(response_text: str) -> list[dict]:
    """Parse a `data: {...}\\n\\n`-delimited SSE body into a list of event dicts."""
    events = []
    for block in response_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))
    return events


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "vector_db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()

    from app.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
