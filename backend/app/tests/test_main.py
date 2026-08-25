def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "has_api": False}


def test_query_without_documents_returns_400(client):
    response = client.post("/api/query", json={"question": "hello?"})
    assert response.status_code == 400


def test_stats_empty(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks"] == 0
    assert body["has_api"] is False
