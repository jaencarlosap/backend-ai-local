from abc import ABC, abstractmethod
from typing import Any, Dict

from src.config import settings


class BaseInferenceEngine(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = settings.device
        self._loaded = False

    @abstractmethod
    def load(self, model_path: str) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def infer(self, input_data: Any, params: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def get_vram_usage_mb(self) -> float:
        pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded
