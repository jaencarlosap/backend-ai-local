import base64
import io
import tempfile
from typing import Any, Dict

from src.inference.base import BaseInferenceEngine
from src.utils.gpu_utils import clear_gpu_cache
from src.utils.logger import logger


class TTSEngine(BaseInferenceEngine):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self._tts = None

    def load(self, model_path: str) -> None:
        from TTS.api import TTS

        logger.info(f"Loading TTS model: {self.model_id}")
        use_gpu = self.device == "cuda"
        self._tts = TTS(model_path=model_path, gpu=use_gpu)
        self._loaded = True
        logger.info(f"TTS model loaded: {self.model_id}")

    def unload(self) -> None:
        del self._tts
        self._tts = None
        self._loaded = False
        clear_gpu_cache()
        logger.info(f"TTS model unloaded: {self.model_id}")

    def infer(self, input_data: Any, params: Dict[str, Any]) -> Any:
        text = str(input_data)
        speaker = params.get("speaker_id")
        speed = params.get("speed", 1.0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            kwargs = {"text": text, "file_path": tmp.name}
            if speaker:
                kwargs["speaker"] = speaker
            if speed != 1.0:
                kwargs["speed"] = speed
            self._tts.tts_to_file(**kwargs)

            with open(tmp.name, "rb") as f:
                audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {"audio_base64": audio_b64, "format": "wav"}

    def get_vram_usage_mb(self) -> float:
        if self._tts is None:
            return 0.0
        # TTS library doesn't expose model params easily; estimate
        return 500.0
