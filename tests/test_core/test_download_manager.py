import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.download_manager import DownloadManager


@pytest.fixture
def download_manager(tmp_path):
    with patch("src.core.download_manager.settings") as mock_settings:
        mock_settings.model_cache_dir = str(tmp_path)
        mock_settings.hf_token = ""
        manager = DownloadManager()
    return manager


def test_is_cached_false(download_manager):
    assert download_manager.is_cached("nonexistent/model") is False


def test_is_cached_true(download_manager):
    path = download_manager._model_disk_path("test/model")
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    assert download_manager.is_cached("test/model") is True


def test_get_disk_path_none(download_manager):
    assert download_manager.get_disk_path("nonexistent/model") is None


def test_get_disk_path_exists(download_manager):
    path = download_manager._model_disk_path("test/model")
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    assert download_manager.get_disk_path("test/model") == str(path)


def test_list_cached_models(download_manager):
    path = download_manager._model_disk_path("org/model")
    path.mkdir(parents=True)
    (path / "weights.bin").write_text("data")
    cached = download_manager.list_cached_models()
    assert len(cached) == 1
    assert cached[0]["model_id"] == "org/model"


def test_model_disk_path_format(download_manager):
    path = download_manager._model_disk_path("org/model-name")
    assert "org--model-name" in str(path)
