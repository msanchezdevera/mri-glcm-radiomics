import pytest
from fastapi.testclient import TestClient

from conftest import SAMPLES


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RADIOMICS_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from mri_radiomics.api import main
    main.get_store.cache_clear()
    main.get_agent.cache_clear()
    return TestClient(main.app)


def upload(client, path, endpoint):
    with open(path, "rb") as f:
        return client.post(endpoint, files={"file": (path.name, f, "image/jpeg")})


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    r = upload(client, SAMPLES / "pituitary.jpg", "/predict")
    assert r.status_code == 200 and r.json()["label"] == "pituitary"


def test_rejects_bad_type(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_case_flow_and_chat(client):
    case = upload(client, SAMPLES / "glioma.jpg", "/cases").json()
    assert client.get(f"/cases/{case['case_id']}").status_code == 200
    r = client.post("/agent/chat", json={"session_id": "t", "message": "Why this result?",
                                         "case_id": case["case_id"]}).json()
    assert r["tool_calls"] == ["predict", "explain_prediction"] and "Glioma" in r["reply"]
    assert client.get("/cases/nope").status_code == 404


def test_ui_and_samples(client):
    assert "MRI Radiomics Agent" in client.get("/").text
    assert client.get("/samples/glioma.jpg").status_code == 200
