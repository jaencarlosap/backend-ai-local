from typing import Dict, Type

from src.inference.base import BaseInferenceEngine
from src.inference.image_engine import ImageEngine
from src.inference.llm_engine import LLMEngine
from src.inference.stt_engine import STTEngine
from src.inference.tts_engine import TTSEngine
from src.inference.video_engine import VideoEngine
from src.models.enums import TaskType

ENGINE_MAP: Dict[TaskType, Type[BaseInferenceEngine]] = {
    TaskType.TEXT: LLMEngine,
    TaskType.AUDIO_TTS: TTSEngine,
    TaskType.AUDIO_STT: STTEngine,
    TaskType.IMAGE: ImageEngine,
    TaskType.VIDEO: VideoEngine,
}


def get_engine_class(task_type: TaskType) -> Type[BaseInferenceEngine]:
    return ENGINE_MAP[task_type]
