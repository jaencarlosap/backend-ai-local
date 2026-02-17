from typing import Any, Dict

import torch

from src.inference.base import BaseInferenceEngine
from src.utils.gpu_utils import clear_gpu_cache
from src.utils.logger import logger


class STTEngine(BaseInferenceEngine):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self._model = None
        self._processor = None

    def load(self, model_path: str) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info(f"Loading STT model: {self.model_id}")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._processor = WhisperProcessor.from_pretrained(model_path)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype
        )
        if self.device == "cuda":
            self._model = self._model.to(self.device)
        self._loaded = True
        logger.info(f"STT model loaded: {self.model_id}")

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._loaded = False
        clear_gpu_cache()
        logger.info(f"STT model unloaded: {self.model_id}")

    def infer(self, input_data: Any, params: Dict[str, Any]) -> Any:
        import soundfile as sf
        import io
        import base64

        language = params.get("language", "en")
        task = params.get("task", "transcribe")

        # input_data should be base64-encoded audio
        audio_bytes = base64.b64decode(input_data)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        input_features = self._processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).input_features.to(self._model.device)

        forced_decoder_ids = self._processor.get_decoder_prompt_ids(
            language=language, task=task
        )

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )

        transcription = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return {"text": transcription[0].strip()}

    def get_vram_usage_mb(self) -> float:
        if self._model is None:
            return 0.0
        return sum(
            p.nelement() * p.element_size() for p in self._model.parameters()
        ) / (1024 ** 2)
