from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.enums import TaskType


class ExecuteRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="HuggingFace model ID or local model name",
        json_schema_extra={"examples": ["gpt2", "openai/whisper-small", "stabilityai/stable-diffusion-2-1"]},
    )
    input: Any = Field(
        ...,
        description=(
            "Input data for the model. For text tasks, a prompt string. "
            "For STT, base64-encoded audio. For image, a text prompt. "
            "For TTS, the text to synthesize."
        ),
        json_schema_extra={"examples": ["The future of AI is"]},
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Task-specific parameters. "
            "Text: max_length, temperature, top_p, do_sample. "
            "Image: guidance_scale, num_inference_steps, width, height. "
            "TTS: speaker, speed. "
            "STT: language, task (transcribe/translate)."
        ),
        json_schema_extra={"examples": [{"max_length": 100, "temperature": 0.7}]},
    )
    force_reload: bool = Field(
        default=False,
        description="Force re-download from HuggingFace and reload the model into VRAM",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_id": "gpt2",
                    "input": "The future of AI is",
                    "params": {"max_length": 100, "temperature": 0.7},
                    "force_reload": False,
                },
                {
                    "model_id": "stabilityai/stable-diffusion-2-1",
                    "input": "A photo of an astronaut riding a horse on the moon",
                    "params": {
                        "guidance_scale": 7.5,
                        "num_inference_steps": 30,
                        "width": 512,
                        "height": 512,
                    },
                    "force_reload": False,
                },
                {
                    "model_id": "Qwen/Qwen3-TTS",
                    "input": "Today is a wonderful day to build something people love.",
                    "params": {"speaker": "Chelsie", "speed": 1.0},
                    "force_reload": False,
                },
                {
                    "model_id": "openai/whisper-small",
                    "input": "<base64-encoded-audio-bytes>",
                    "params": {"language": "en", "task": "transcribe"},
                    "force_reload": False,
                },
            ]
        }
    }


class ExecuteResponse(BaseModel):
    model_id: str = Field(..., description="The model that performed the inference")
    task_type: str = Field(..., description="The task type that was executed")
    result: Any = Field(
        ...,
        description=(
            "Inference result. Structure varies by task type: "
            "text -> {generated_text}, image -> {image_base64, format}, "
            "tts -> {audio_base64, format, sample_rate}, stt -> {text}"
        ),
    )
    vram_usage_percent: float = Field(
        ..., description="Current GPU VRAM usage as a percentage (0-100)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_id": "gpt2",
                    "task_type": "text",
                    "result": {"generated_text": "The future of AI is bright and full of possibilities..."},
                    "vram_usage_percent": 34.5,
                },
                {
                    "model_id": "stabilityai/stable-diffusion-2-1",
                    "task_type": "image",
                    "result": {"image_base64": "<base64-encoded-png>", "format": "png"},
                    "vram_usage_percent": 72.1,
                },
                {
                    "model_id": "Qwen/Qwen3-TTS",
                    "task_type": "audio_tts",
                    "result": {"audio_base64": "<base64-encoded-wav>", "format": "wav", "sample_rate": 24000},
                    "vram_usage_percent": 41.3,
                },
                {
                    "model_id": "openai/whisper-small",
                    "task_type": "audio_stt",
                    "result": {"text": "Hello, how are you doing today?"},
                    "vram_usage_percent": 28.7,
                },
            ]
        }
    }


class ModelStatusItem(BaseModel):
    model_id: str = Field(..., description="HuggingFace model identifier")
    task_type: Optional[str] = Field(None, description="Associated task type (text, image, audio_tts, audio_stt, video)")
    state: str = Field(..., description="Current state: downloading, on_disk, loaded, or failed")
    vram_mb: float = Field(0.0, description="VRAM consumed by this model in megabytes (0 if not loaded)")
    last_used: Optional[float] = Field(None, description="Unix timestamp of last inference call")


class ModelStatusResponse(BaseModel):
    models: List[ModelStatusItem] = Field(..., description="List of all known models (loaded + cached on disk)")
    vram_usage_percent: float = Field(..., description="Current GPU VRAM usage as a percentage (0-100)")
    active_downloads: List[str] = Field(..., description="Model IDs currently being downloaded")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "models": [
                        {
                            "model_id": "gpt2",
                            "task_type": "text",
                            "state": "loaded",
                            "vram_mb": 548.0,
                            "last_used": 1708099200.0,
                        },
                        {
                            "model_id": "stabilityai/stable-diffusion-2-1",
                            "task_type": "image",
                            "state": "on_disk",
                            "vram_mb": 0.0,
                            "last_used": None,
                        },
                    ],
                    "vram_usage_percent": 34.5,
                    "active_downloads": [],
                }
            ]
        }
    }


class FetchRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="HuggingFace model ID to pre-download to local disk cache without loading into VRAM",
        json_schema_extra={"examples": ["gpt2"]},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"model_id": "gpt2"},
                {"model_id": "stabilityai/stable-diffusion-2-1"},
                {"model_id": "Qwen/Qwen3-TTS"},
                {"model_id": "openai/whisper-small"},
            ]
        }
    }


class FetchResponse(BaseModel):
    model_id: str = Field(..., description="The fetched model identifier")
    path: str = Field(..., description="Local disk path where the model is stored")
    message: str = Field(..., description="Human-readable status message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_id": "gpt2",
                    "path": "/app/models/gpt2",
                    "message": "Model gpt2 is available on disk",
                },
                {
                    "model_id": "Qwen/Qwen3-TTS",
                    "path": "/app/models/Qwen--Qwen3-TTS",
                    "message": "Model Qwen/Qwen3-TTS is available on disk",
                },
            ]
        }
    }


class PurgeResponse(BaseModel):
    message: str = Field(..., description="Confirmation message")

    model_config = {
        "json_schema_extra": {
            "examples": [{"message": "All models purged from VRAM"}]
        }
    }


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error description")

    model_config = {
        "json_schema_extra": {
            "examples": [{"detail": "Model not found: nonexistent/model"}]
        }
    }
