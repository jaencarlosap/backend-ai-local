import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.vram_manager import VRAMManager
from src.models.enums import ModelState, TaskType
from src.models.model_info import ModelInfo


@pytest.fixture
def vram_manager():
    with patch("src.core.vram_manager.settings") as mock_settings:
        mock_settings.vram_threshold = 0.9
        mock_settings.max_vram_gb = 8.0
        manager = VRAMManager()
    return manager


def _make_model_info(model_id: str, vram_mb: float = 500, last_used: float = None):
    engine = MagicMock()
    engine.unload = MagicMock()
    info = ModelInfo(
        model_id=model_id,
        task_type=TaskType.TEXT,
        state=ModelState.LOADED,
        vram_mb=vram_mb,
        last_used=last_used or time.time(),
        engine_instance=engine,
    )
    return info


def test_register_and_loaded_models(vram_manager):
    info = _make_model_info("gpt2")
    vram_manager.register_model(info)
    assert "gpt2" in vram_manager.loaded_models


def test_unregister_model(vram_manager):
    info = _make_model_info("gpt2")
    vram_manager.register_model(info)
    vram_manager.unregister_model("gpt2")
    assert "gpt2" not in vram_manager.loaded_models
    info.engine_instance.unload.assert_called_once()


def test_update_access_time(vram_manager):
    info = _make_model_info("gpt2", last_used=100.0)
    vram_manager.register_model(info)
    vram_manager.update_access_time("gpt2")
    assert vram_manager.loaded_models["gpt2"].last_used > 100.0


@patch("src.core.vram_manager.get_vram_usage_gb", return_value=5.0)
@patch("src.core.vram_manager.get_vram_total_gb", return_value=8.0)
def test_can_load_model(mock_total, mock_usage, vram_manager):
    # Limit = 8.0 * 0.9 = 7.2, current = 5.0, so 2.0 should fit
    assert vram_manager.can_load_model(2.0) is True
    assert vram_manager.can_load_model(3.0) is False


@patch("src.core.vram_manager.get_vram_usage_gb", return_value=6.5)
@patch("src.core.vram_manager.get_vram_total_gb", return_value=8.0)
@patch("src.core.vram_manager.clear_gpu_cache")
def test_evict_lru_order(mock_cache, mock_total, mock_usage, vram_manager):
    old = _make_model_info("old_model", last_used=100.0)
    new = _make_model_info("new_model", last_used=999.0)
    vram_manager.register_model(old)
    vram_manager.register_model(new)

    # After eviction, the oldest should be removed first
    # Mock can_load to return True after one eviction
    call_count = 0
    original = vram_manager.can_load_model

    def side_effect(gb):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            return True
        return False

    with patch.object(vram_manager, "can_load_model", side_effect=side_effect):
        vram_manager.evict_lru(1.0)

    assert "old_model" not in vram_manager.loaded_models
    assert "new_model" in vram_manager.loaded_models


def test_purge_all(vram_manager):
    with patch("src.core.vram_manager.clear_gpu_cache"):
        info1 = _make_model_info("m1")
        info2 = _make_model_info("m2")
        vram_manager.register_model(info1)
        vram_manager.register_model(info2)
        vram_manager.purge_all()
        assert len(vram_manager.loaded_models) == 0
