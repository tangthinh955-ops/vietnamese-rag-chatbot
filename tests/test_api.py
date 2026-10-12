import pytest
from fastapi.testclient import TestClient

from rag.api.main import create_app


@pytest.fixture()
def client(pipeline):
    app = create_app()
    app.state.pipeline = pipeline  # inject the test pipeline before lifespan
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["chunks"] > 0


def test_stats(client):
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["mock_llm"] is True
    assert body["chunks"] > 0


def test_chat(client):
    r = client.post("/chat", json={"question": "Đổi trả trong bao nhiêu ngày?"})
    assert r.status_code == 200
    body = r.json()
    assert "7 ngày" in body["answer"]
    assert body["sources"][0]["source"]
    assert body["session_id"]


def test_chat_with_session(client):
    r1 = client.post("/chat", json={"question": "Freeship bao nhiêu?"})
    sid = r1.json()["session_id"]
    r2 = client.post(
        "/chat", json={"question": "Còn son môi thì sao?", "session_id": sid}
    )
    assert r2.status_code == 200
    assert r2.json()["session_id"] == sid


def test_chat_stream(client):
    with client.stream("GET", "/chat/stream", params={"question": "Son giá?"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert "event: token" in body
    assert "event: done" in body


def test_ingest_files_upload(client, tmp_path):
    r = client.post(
        "/ingest/files",
        files=[("files", ("note.md", b"# Voucher\n\nMua 2 tang 1.", "text/markdown"))],
    )
    assert r.status_code == 200
    assert r.json()["chunks_added"] > 0


def test_ingest_rejects_unsupported(client):
    r = client.post(
        "/ingest/files",
        files=[("files", ("x.exe", b"bad", "application/octet-stream"))],
    )
    assert r.status_code == 400
