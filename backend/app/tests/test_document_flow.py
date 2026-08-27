import io

from app.tests.conftest import parse_sse


def _upload_sample(client, name="sample.txt", text=None):
    text = text or (
        "Machine learning is a subset of artificial intelligence.\n\n"
        "It enables systems to learn from data without explicit programming.\n\n"
        "Popular frameworks include PyTorch and TensorFlow."
    )
    return client.post(
        "/api/documents",
        files={"file": (name, io.BytesIO(text.encode()), "text/plain")},
    )


def test_upload_query_history_delete_flow(client):
    upload_resp = _upload_sample(client)
    assert upload_resp.status_code == 200
    body = upload_resp.json()
    assert body["success"] is True
    assert body["chunks_created"] >= 1

    stats_resp = client.get("/api/stats")
    assert stats_resp.status_code == 200
    assert stats_resp.json()["total_chunks"] == body["chunks_created"]

    query_resp = client.post("/api/query", json={"question": "What is machine learning?"})
    assert query_resp.status_code == 200
    assert query_resp.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(query_resp.text)
    assert [e["type"] for e in events] == ["sources", "token", "done"]

    sources_event, token_event, done_event = events
    assert sources_event["retrieved_chunks"] >= 1
    assert len(sources_event["sources"]) >= 1
    assert token_event["text"]
    assert done_event["model"] == "demo"

    history_resp = client.get("/api/history")
    assert history_resp.status_code == 200
    assert len(history_resp.json()) == 1

    clear_history_resp = client.delete("/api/history")
    assert clear_history_resp.status_code == 200
    assert client.get("/api/history").json() == []

    delete_resp = client.delete(f"/api/documents/{upload_resp.json()['filename']}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted_chunks"] == body["chunks_created"]

    assert client.get("/api/stats").json()["total_chunks"] == 0


def test_upload_rejects_unsupported_extension(client):
    resp = client.post(
        "/api/documents",
        files={"file": ("sample.exe", io.BytesIO(b"binary"), "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_delete_unknown_source_returns_404(client):
    resp = client.delete("/api/documents/does-not-exist.txt")
    assert resp.status_code == 404


def test_documents_are_isolated_between_sessions(client):
    upload_resp = _upload_sample(client, name="session-a.txt")
    assert upload_resp.status_code == 200
    session_a_cookies = dict(client.cookies)

    # Simulate a second browser session on the same server by dropping the
    # session cookie - the next request gets a brand new session_id.
    client.cookies.clear()

    stats_as_session_b = client.get("/api/stats")
    assert stats_as_session_b.json()["total_chunks"] == 0

    query_as_session_b = client.post("/api/query", json={"question": "anything?"})
    assert query_as_session_b.status_code == 400  # no documents in this session

    # Switching back to session A's cookie should see its document again.
    client.cookies.update(session_a_cookies)
    stats_as_session_a = client.get("/api/stats")
    assert stats_as_session_a.json()["total_chunks"] >= 1
    assert "session-a.txt" in stats_as_session_a.json()["sources"]
