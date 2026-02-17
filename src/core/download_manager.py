from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set

from huggingface_hub import snapshot_download

from src.config import settings
from src.models.enums import ModelState
from src.utils.exceptions import DownloadError
from src.utils.logger import logger


class DownloadManager:
    def __init__(self):
        self._cache_dir = Path(settings.model_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._active_downloads: Set[str] = set()

    def _model_disk_path(self, model_id: str) -> Path:
        safe_name = model_id.replace("/", "--")
        return self._cache_dir / safe_name

    def is_cached(self, model_id: str) -> bool:
        path = self._model_disk_path(model_id)
        return path.exists() and any(path.iterdir())

    def get_disk_path(self, model_id: str) -> Optional[str]:
        path = self._model_disk_path(model_id)
        if path.exists() and any(path.iterdir()):
            return str(path)
        return None

    def _download_sync(self, model_id: str) -> str:
        token = settings.hf_token or None
        local_dir = self._model_disk_path(model_id)
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                token=token,
            )
            return str(local_dir)
        except Exception as e:
            raise DownloadError(model_id, str(e))

    async def download(self, model_id: str) -> str:
        if self.is_cached(model_id):
            logger.info(f"Model {model_id} found in disk cache")
            return str(self._model_disk_path(model_id))

        if model_id in self._active_downloads:
            logger.info(f"Download already in progress for {model_id}")
            while model_id in self._active_downloads:
                await asyncio.sleep(1)
            return str(self._model_disk_path(model_id))

        self._active_downloads.add(model_id)
        logger.info(f"Starting download: {model_id}")
        try:
            loop = asyncio.get_event_loop()
            path = await loop.run_in_executor(
                self._executor, self._download_sync, model_id
            )
            logger.info(f"Download complete: {model_id}")
            return path
        finally:
            self._active_downloads.discard(model_id)

    def list_cached_models(self) -> List[Dict[str, str]]:
        results = []
        if not self._cache_dir.exists():
            return results
        for entry in self._cache_dir.iterdir():
            if entry.is_dir():
                model_id = entry.name.replace("--", "/")
                results.append({
                    "model_id": model_id,
                    "path": str(entry),
                    "state": ModelState.ON_DISK.value,
                })
        return results

    @property
    def active_downloads(self) -> Set[str]:
        return set(self._active_downloads)
