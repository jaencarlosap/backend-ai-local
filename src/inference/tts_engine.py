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
        self._qwen_model = None

    def _is_qwen_tts(self) -> bool:
        model_lower = self.model_id.lower()
        return "qwen" in model_lower and "tts" in model_lower

    def load(self, model_path: str) -> None:
        if self._is_qwen_tts():
            self._load_qwen(model_path)
        else:
            self._load_coqui(model_path)
        self._loaded = True

    def _load_qwen(self, model_path: str) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        logger.info(f"Loading Qwen3-TTS model: {self.model_id}")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._qwen_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=self.device if self.device == "cuda" else None,
            dtype=dtype,
        )
        logger.info(f"Qwen3-TTS model loaded: {self.model_id}")

    def _load_coqui(self, model_path: str) -> None:
        from TTS.api import TTS

        logger.info(f"Loading TTS model: {self.model_id}")
        use_gpu = self.device == "cuda"
        self._tts = TTS(model_path=model_path, gpu=use_gpu)
        logger.info(f"TTS model loaded: {self.model_id}")

    def unload(self) -> None:
        del self._tts
        self._tts = None
        del self._qwen_model
        self._qwen_model = None
        self._loaded = False
        clear_gpu_cache()
        logger.info(f"TTS model unloaded: {self.model_id}")

    def infer(self, input_data: Any, params: Dict[str, Any]) -> Any:
        text = str(input_data)
        if self._is_qwen_tts():
            return self._infer_qwen(text, params)
        return self._infer_coqui(text, params)

    def _infer_qwen(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        import soundfile as sf

        language = params.get("language", "Auto")
        speaker = params.get("speaker", "Chelsie")

        wavs, sr = self._qwen_model.generate_custom_voice(
            text=text, language=language, speaker=speaker,
        )

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"audio_base64": audio_b64, "format": "wav", "sample_rate": sr}

    def _infer_coqui(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        speaker = params.get("speaker") or params.get("speaker_id")
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
        if self._is_qwen_tts():
            if self._qwen_model is None:
                return 0.0
            total_params = sum(
                p.numel() for p in self._qwen_model.parameters()
            )
            # bfloat16 = 2 bytes per param
            return (total_params * 2) / (1024 * 1024)
        if self._tts is None:
            return 0.0
        return 500.0
