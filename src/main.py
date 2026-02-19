from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from src.api.v1 import execute as execute_module
from src.api.v1 import models as models_module
from src.config import settings
from src.core.model_manager import ModelManager
from src.utils.gpu_utils import initialize_gpu
from src.utils.logger import logger

model_manager: ModelManager = None  # type: ignore

TAGS_METADATA = [
    {
        "name": "health",
        "description": "System health and readiness checks.",
    },
    {
        "name": "execute",
        "description": (
            "Run AI inference tasks. Supports **text generation**, **image generation**, "
            "**text-to-speech**, **speech-to-text**, and **video** (placeholder). "
            "Models are automatically downloaded from HuggingFace on first use and "
            "managed in GPU VRAM with LRU eviction."
        ),
    },
    {
        "name": "models",
        "description": (
            "Manage the model lifecycle — view loaded/cached models, "
            "pre-download models to disk, and purge GPU VRAM."
        ),
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager
    logger.info("DAMO starting up...")
    device = initialize_gpu()
    settings.device = device

    model_manager = ModelManager()
    execute_module.model_manager = model_manager
    models_module.model_manager = model_manager

    logger.info(f"DAMO ready on device={device}")
    yield

    logger.info("DAMO shutting down — purging models...")
    await model_manager.purge_all()
    logger.info("DAMO shutdown complete")


app = FastAPI(
    title="DAMO — Dynamic AI Model Orchestrator",
    version="0.1.0",
    summary="Unified REST API for multimodal AI inference on consumer-grade GPUs",
    description=(
        "## Overview\n\n"
        "DAMO is a self-hosted API server that provides a single entry point for running "
        "multiple AI tasks — text generation, image generation, speech-to-text, and "
        "text-to-speech — on consumer-grade gaming hardware.\n\n"
        "## Key Features\n\n"
        "- **On-demand model loading** — Models are downloaded from HuggingFace automatically "
        "on first inference request and cached to disk.\n"
        "- **Intelligent VRAM management** — LRU eviction ensures your GPU never runs out of "
        "memory. When a new model needs space, the least-recently-used model is unloaded.\n"
        "- **Multi-task support** — Text (GPT-2, LLaMA, etc.), Images (Stable Diffusion), "
        "Audio TTS (Coqui), Audio STT (Whisper), Video (planned).\n"
        "- **Pre-warming** — Download models ahead of time via `/v1/models/fetch` so inference "
        "calls only need to load from disk.\n\n"
        "## Quick Start\n\n"
        "```bash\n"
        "# Health check\n"
        "curl http://localhost:8000/health\n\n"
        "# Generate text with GPT-2\n"
        'curl -X POST http://localhost:8000/v1/execute/text \\\n'
        '  -H "Content-Type: application/json" \\\n'
        "  -d '{\"model_id\":\"gpt2\",\"input\":\"The future of AI is\","
        "\"params\":{\"max_length\":50}}'\n\n"
        "# Check loaded models\n"
        "curl http://localhost:8000/v1/models/status\n\n"
        "# Free GPU memory\n"
        "curl -X DELETE http://localhost:8000/v1/models/purge\n"
        "```\n"
    ),
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
    docs_url=None,
    redoc_url="/docs",
    openapi_url="/openapi.json",
)

app.include_router(execute_module.router, prefix="/v1", tags=["execute"])
app.include_router(models_module.router, prefix="/v1", tags=["models"])


@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description=(
        "Returns the current health status of the DAMO server, including "
        "the active compute device (cuda/cpu) and current VRAM usage percentage."
    ),
    response_description="Server health status",
    responses={
        200: {
            "description": "Server is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "device": "cuda",
                        "vram_usage_percent": 34.5,
                    }
                }
            },
        }
    },
)
async def health():
    return {
        "status": "ok",
        "device": settings.device,
        "vram_usage_percent": (
            model_manager.vram_manager.get_vram_usage_percent()
            if model_manager
            else 0.0
        ),
    }
