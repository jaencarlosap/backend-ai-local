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
            "For STT, base64-encoded audio. For image, a text prompt."
        ),
        json_schema_extra={"examples": ["The future of AI is"]},
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Task-specific parameters. "
            "Text: max_length, temperature, top_p, do_sample. "
            "Image: guidance_scale, num_inference_steps, width, height. "
            "TTS: speaker_id, speed. "
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
                }
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
            "tts -> {audio_base64, format}, stt -> {text}"
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
                }
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
                        }
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
            "examples": [{"model_id": "gpt2"}]
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
                }
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
