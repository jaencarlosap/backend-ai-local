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
        "- `audio_tts` — Text-to-Speech (e.g. Qwen3-TTS). Params: `speaker`, `speed`\n"
        "- `audio_stt` — Speech-to-Text (e.g. Whisper). Input: base64 audio. Params: `language`, `task`\n"
        "- `video` — Video generation (not yet implemented)\n\n"
        "**VRAM management:** If GPU memory is insufficient, the least-recently-used model "
        "is automatically evicted to make room.\n\n"
        "---\n\n"
        "### Examples\n\n"
        "**Text generation (GPT-2):**\n"
        "```bash\n"
        'curl -X POST http://localhost:8000/v1/execute/text \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"model_id":"gpt2","input":"The future of AI is","params":{"max_length":100,"temperature":0.7}}\'\n'
        "```\n\n"
        "**Image generation (Stable Diffusion):**\n"
        "```bash\n"
        'curl -X POST http://localhost:8000/v1/execute/image \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"model_id":"stabilityai/stable-diffusion-2-1","input":"A photo of an astronaut riding a horse on the moon","params":{"guidance_scale":7.5,"num_inference_steps":30,"width":512,"height":512}}\'\n'
        "```\n\n"
        "**Text-to-Speech (Qwen3-TTS):**\n"
        "```bash\n"
        'curl -X POST http://localhost:8000/v1/execute/audio_tts \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"model_id":"Qwen/Qwen3-TTS","input":"Today is a wonderful day to build something people love.","params":{"speaker":"Chelsie","speed":1.0}}\'\n'
        "```\n\n"
        "**Speech-to-Text (Whisper):**\n"
        "```bash\n"
        'curl -X POST http://localhost:8000/v1/execute/audio_stt \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"model_id":"openai/whisper-small","input":"<base64-encoded-audio>","params":{"language":"en","task":"transcribe"}}\'\n'
        "```"
    ),
    response_description="Inference result with model metadata and current VRAM usage",
    responses={
        200: {
            "description": "Inference completed successfully",
            "model": ExecuteResponse,
            "content": {
                "application/json": {
                    "examples": {
                        "text_generation": {
                            "summary": "Text generation (GPT-2)",
                            "value": {
                                "model_id": "gpt2",
                                "task_type": "text",
                                "result": {"generated_text": "The future of AI is bright and full of possibilities..."},
                                "vram_usage_percent": 34.5,
                            },
                        },
                        "image_generation": {
                            "summary": "Image generation (Stable Diffusion)",
                            "value": {
                                "model_id": "stabilityai/stable-diffusion-2-1",
                                "task_type": "image",
                                "result": {"image_base64": "<base64-encoded-png>", "format": "png"},
                                "vram_usage_percent": 72.1,
                            },
                        },
                        "tts": {
                            "summary": "Text-to-Speech (Qwen3-TTS)",
                            "value": {
                                "model_id": "Qwen/Qwen3-TTS",
                                "task_type": "audio_tts",
                                "result": {"audio_base64": "<base64-encoded-wav>", "format": "wav", "sample_rate": 24000},
                                "vram_usage_percent": 41.3,
                            },
                        },
                        "stt": {
                            "summary": "Speech-to-Text (Whisper)",
                            "value": {
                                "model_id": "openai/whisper-small",
                                "task_type": "audio_stt",
                                "result": {"text": "Hello, how are you doing today?"},
                                "vram_usage_percent": 28.7,
                            },
                        },
                    }
                }
            },
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
