from enum import Enum


class TaskType(str, Enum):
    TEXT = "text"
    AUDIO_TTS = "audio_tts"
    AUDIO_STT = "audio_stt"
    IMAGE = "image"
    VIDEO = "video"


class ModelState(str, Enum):
    DOWNLOADING = "downloading"
    ON_DISK = "on_disk"
    LOADED = "loaded"
    FAILED = "failed"
