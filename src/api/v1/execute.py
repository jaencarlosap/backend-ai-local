from fastapi import APIRouter, HTTPException

from src.api.v1.schemas import ErrorResponse, ExecuteRequest, ExecuteResponse
from src.core.model_manager import ModelManager
from src.models.enums import TaskType
from src.utils.exceptions import (
    DownloadError,
    InvalidParametersError,
    ModelNotFoundError,
    VRAMExhaustedError,
)

router = APIRouter()

# Injected at startup
model_manager: ModelManager = None  # type: ignore


@router.post(
    "/execute/{task_type}",
    response_model=ExecuteResponse,
    summary="Execute AI inference",
    description=(
        "Run inference on a specified model for a given task type. "
        "The model is automatically downloaded from HuggingFace if not cached locally, "
        "loaded into GPU VRAM (with LRU eviction if needed), and then used for inference.\n\n"
        "**Supported task types:**\n"
        "- `text` — Text generation (e.g. GPT-2, LLaMA). Params: `max_length`, `temperature`, `top_p`, `do_sample`\n"
        "- `image` — Image generation (e.g. Stable Diffusion). Params: `guidance_scale`, `num_inference_steps`, `width`, `height`\n"
        "- `audio_tts` — Text-to-Speech (Coqui TTS). Params: `speaker_id`, `speed`\n"
        "- `audio_stt` — Speech-to-Text (Whisper). Input: base64 audio. Params: `language`, `task`\n"
        "- `video` — Video generation (not yet implemented)\n\n"
        "**VRAM management:** If GPU memory is insufficient, the least-recently-used model "
        "is automatically evicted to make room."
    ),
    response_description="Inference result with model metadata and current VRAM usage",
    responses={
        200: {
            "description": "Inference completed successfully",
            "model": ExecuteResponse,
        },
        404: {
            "description": "Model not found on HuggingFace or download failed",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Failed to download nonexistent/model: 404 Client Error"}
                }
            },
        },
        422: {
            "description": "Invalid parameters for the given task type",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid parameters: max_length must be a positive integer"}
                }
            },
        },
        501: {
            "description": "Task type not yet implemented (e.g. video)",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Video generation requires 24GB+ VRAM and is not yet supported"}
                }
            },
        },
        507: {
            "description": "Insufficient VRAM even after LRU eviction",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Not enough VRAM: need 4.0GB, available 1.2GB"}
                }
            },
        },
    },
)
async def execute_task(task_type: TaskType, request: ExecuteRequest):
    try:
        result = await model_manager.infer(
            model_id=request.model_id,
            task_type=task_type,
            input_data=request.input,
            params=request.params,
            force_reload=request.force_reload,
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except VRAMExhaustedError as e:
        raise HTTPException(status_code=507, detail=str(e))
    except InvalidParametersError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except DownloadError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ExecuteResponse(
        model_id=request.model_id,
        task_type=task_type.value,
        result=result,
        vram_usage_percent=model_manager.vram_manager.get_vram_usage_percent(),
    )
