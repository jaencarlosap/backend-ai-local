from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Patch model_manager before importing app
    mock_mm = MagicMock()
    mock_mm.vram_manager.get_vram_usage_percent.return_value = 25.0
    mock_mm.get_all_model_status.return_value = []
    mock_mm.download_manager.active_downloads = set()
    mock_mm.purge_all = AsyncMock()
    mock_mm.fetch_model = AsyncMock(return_value="/app/models/gpt2")
    mock_mm.infer = AsyncMock(return_value={"generated_text": "hello world"})

    with patch("src.api.v1.execute.model_manager", mock_mm), \
         patch("src.api.v1.models.model_manager", mock_mm), \
         patch("src.main.model_manager", mock_mm):
        from src.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_execute_text(client):
    resp = client.post(
        "/v1/execute/text",
        json={"model_id": "gpt2", "input": "Hello", "params": {"max_length": 50}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "gpt2"
    assert data["task_type"] == "text"
    assert "result" in data


def test_models_status(client):
    resp = client.get("/v1/models/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "vram_usage_percent" in data


def test_models_purge(client):
    resp = client.delete("/v1/models/purge")
    assert resp.status_code == 200
    assert "purged" in resp.json()["message"].lower()


def test_models_fetch(client):
    resp = client.post(
        "/v1/models/fetch",
        json={"model_id": "gpt2"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "gpt2"


def test_invalid_task_type(client):
    resp = client.post(
        "/v1/execute/invalid_type",
        json={"model_id": "gpt2", "input": "Hello"},
    )
    assert resp.status_code == 422
