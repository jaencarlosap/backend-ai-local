from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from src.core.download_manager import DownloadManager
from src.core.task_router import get_engine_class
from src.core.vram_manager import VRAMManager
from src.models.enums import ModelState, TaskType
from src.models.model_info import ModelInfo
from src.utils.logger import logger


class ModelManager:
    def __init__(self):
        self.vram_manager = VRAMManager()
        self.download_manager = DownloadManager()
        self._lock = asyncio.Lock()
        self._model_registry: Dict[str, ModelInfo] = {}

    async def load_model(
        self,
        model_id: str,
        task_type: TaskType,
        force_reload: bool = False,
    ) -> ModelInfo:
        async with self._lock:
            existing = self._model_registry.get(model_id)
            if existing and existing.state == ModelState.LOADED and not force_reload:
                existing.touch()
                self.vram_manager.update_access_time(model_id)
                return existing

            if force_reload and existing and existing.state == ModelState.LOADED:
                self.vram_manager.unregister_model(model_id)

            # Download if needed
            info = ModelInfo(model_id=model_id, task_type=task_type)
            info.state = ModelState.DOWNLOADING
            self._model_registry[model_id] = info

        # Download outside lock
        disk_path = await self.download_manager.download(model_id)

        async with self._lock:
            info = self._model_registry[model_id]
            info.disk_path = disk_path
            info.state = ModelState.ON_DISK

            # Create engine and estimate VRAM
            engine_cls = get_engine_class(task_type)
            engine = engine_cls(model_id)

            # Evict if needed (estimate ~2GB if unknown)
            required_gb = 2.0
            if not self.vram_manager.can_load_model(required_gb):
                self.vram_manager.evict_lru(required_gb)

            # Load model (blocking — run in thread)
            try:
                await asyncio.to_thread(engine.load, disk_path)
            except Exception as e:
                info.state = ModelState.FAILED
                logger.error(f"Failed to load {model_id}: {e}")
                raise

            info.engine_instance = engine
            info.vram_mb = engine.get_vram_usage_mb()
            info.state = ModelState.LOADED
            info.touch()
            self.vram_manager.register_model(info)
            return info

    async def infer(
        self,
        model_id: str,
        task_type: TaskType,
        input_data: Any,
        params: Dict[str, Any],
        force_reload: bool = False,
    ) -> Any:
        info = await self.load_model(model_id, task_type, force_reload)
        result = await asyncio.to_thread(
            info.engine_instance.infer, input_data, params
        )
        self.vram_manager.update_access_time(model_id)
        return result

    def get_all_model_status(self) -> list:
        statuses = []
        for model_id, info in self._model_registry.items():
            statuses.append({
                "model_id": info.model_id,
                "task_type": info.task_type.value,
                "state": info.state.value,
                "vram_mb": info.vram_mb,
                "last_used": info.last_used,
            })
        # Include cached-only models
        for cached in self.download_manager.list_cached_models():
            if cached["model_id"] not in self._model_registry:
                statuses.append(cached)
        return statuses

    async def purge_all(self):
        async with self._lock:
            self.vram_manager.purge_all()
            self._model_registry.clear()
            logger.info("All models purged")

    async def fetch_model(self, model_id: str) -> str:
        return await self.download_manager.download(model_id)
