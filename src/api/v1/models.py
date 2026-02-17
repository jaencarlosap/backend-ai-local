from fastapi import APIRouter, HTTPException

from src.api.v1.schemas import (
    ErrorResponse,
    FetchRequest,
    FetchResponse,
    ModelStatusItem,
    ModelStatusResponse,
    PurgeResponse,
)
from src.core.model_manager import ModelManager
from src.utils.exceptions import DownloadError

router = APIRouter()

# Injected at startup
model_manager: ModelManager = None  # type: ignore


@router.get(
    "/models/status",
    response_model=ModelStatusResponse,
    summary="Get all models status",
    description=(
        "Returns the status of every model known to the system, including:\n\n"
        "- Models currently **loaded** in GPU VRAM\n"
        "- Models **cached on disk** but not loaded\n"
        "- Models currently **being downloaded**\n\n"
        "Also reports overall VRAM usage percentage and any active downloads."
    ),
    response_description="List of all models with their state and VRAM metrics",
)
async def get_models_status():
    statuses = model_manager.get_all_model_status()
    items = [ModelStatusItem(**s) for s in statuses]
    return ModelStatusResponse(
        models=items,
        vram_usage_percent=model_manager.vram_manager.get_vram_usage_percent(),
        active_downloads=list(model_manager.download_manager.active_downloads),
    )


@router.delete(
    "/models/purge",
    response_model=PurgeResponse,
    summary="Purge all loaded models",
    description=(
        "Unloads **all** models from GPU VRAM and clears the CUDA cache. "
        "Models remain cached on disk and can be reloaded on the next inference request. "
        "Use this to free up GPU memory without deleting downloaded model files."
    ),
    response_description="Confirmation that all models have been purged from VRAM",
)
async def purge_models():
    await model_manager.purge_all()
    return PurgeResponse(message="All models purged from VRAM")


@router.post(
    "/models/fetch",
    response_model=FetchResponse,
    summary="Pre-download a model",
    description=(
        "Downloads a model from HuggingFace to the local disk cache **without** loading it "
        "into GPU VRAM. This is useful for pre-warming: download large models ahead of time "
        "so that subsequent `/v1/execute` calls only need to load from disk.\n\n"
        "If the model is already cached, returns immediately with the existing path."
    ),
    response_description="Download confirmation with the local disk path",
    responses={
        404: {
            "description": "Model not found on HuggingFace",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Failed to download nonexistent/model: 404 Client Error"}
                }
            },
        },
    },
)
async def fetch_model(request: FetchRequest):
    try:
        path = await model_manager.fetch_model(request.model_id)
    except DownloadError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return FetchResponse(
        model_id=request.model_id,
        path=path,
        message=f"Model {request.model_id} is available on disk",
    )
