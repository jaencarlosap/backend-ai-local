from typing import Any, Dict

from src.inference.base import BaseInferenceEngine
from src.utils.logger import logger


class VideoEngine(BaseInferenceEngine):
    def __init__(self, model_id: str):
        super().__init__(model_id)

    def load(self, model_path: str) -> None:
        raise NotImplementedError(
            "Video generation requires 24GB+ VRAM and is not yet supported"
        )

    def unload(self) -> None:
        self._loaded = False

    def infer(self, input_data: Any, params: Dict[str, Any]) -> Any:
        raise NotImplementedError("Video inference is not yet supported")

    def get_vram_usage_mb(self) -> float:
        return 0.0
