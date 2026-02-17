from __future__ import annotations

from typing import Dict

from src.config import settings
from src.models.model_info import ModelInfo
from src.utils.exceptions import VRAMExhaustedError
from src.utils.gpu_utils import clear_gpu_cache, get_vram_total_gb, get_vram_usage_gb
from src.utils.logger import logger


class VRAMManager:
    def __init__(self):
        self._loaded_models: Dict[str, ModelInfo] = {}
        self._threshold = settings.vram_threshold
        self._max_vram_gb = settings.max_vram_gb

    @property
    def loaded_models(self) -> Dict[str, ModelInfo]:
        return dict(self._loaded_models)

    def get_effective_limit_gb(self) -> float:
        total = get_vram_total_gb()
        if total > 0:
            return total * self._threshold
        return self._max_vram_gb * self._threshold

    def can_load_model(self, required_gb: float) -> bool:
        current = get_vram_usage_gb()
        limit = self.get_effective_limit_gb()
        return (current + required_gb) <= limit

    def register_model(self, info: ModelInfo):
        self._loaded_models[info.model_id] = info
        logger.info(
            f"Registered model {info.model_id} "
            f"(VRAM: {info.vram_mb:.0f}MB)"
        )

    def unregister_model(self, model_id: str):
        info = self._loaded_models.pop(model_id, None)
        if info and info.engine_instance:
            info.engine_instance.unload()
            clear_gpu_cache()
            logger.info(f"Unloaded model {model_id}")

    def evict_lru(self, required_gb: float):
        sorted_models = sorted(
            self._loaded_models.values(), key=lambda m: m.last_used
        )
        for model_info in sorted_models:
            if self.can_load_model(required_gb):
                return
            logger.info(f"Evicting LRU model: {model_info.model_id}")
            self.unregister_model(model_info.model_id)

        if not self.can_load_model(required_gb):
            available = self.get_effective_limit_gb() - get_vram_usage_gb()
            raise VRAMExhaustedError(required_gb, max(0, available))

    def update_access_time(self, model_id: str):
        if model_id in self._loaded_models:
            self._loaded_models[model_id].touch()

    def get_vram_usage_percent(self) -> float:
        total = get_vram_total_gb()
        if total == 0:
            return 0.0
        return get_vram_usage_gb() / total * 100

    def purge_all(self):
        model_ids = list(self._loaded_models.keys())
        for model_id in model_ids:
            self.unregister_model(model_id)
        clear_gpu_cache()
        logger.info("All models purged from VRAM")
