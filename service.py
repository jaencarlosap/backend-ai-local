import bentoml
from src.config import settings
from src.core.model_manager import ModelManager
from src.models.enums import TaskType
from src.utils.gpu_utils import initialize_gpu
from src.utils.logger import logger
from src.schemas import (
    ExecuteRequest, ExecuteResponse,
    ModelStatusResponse, ModelStatusItem,
    FetchRequest, FetchResponse, PurgeResponse,
)


@bentoml.service(
    name="damo",
    resources={"gpu": 1},
    traffic={"timeout": 300},
)
class DAMOService:
    def __init__(self):
        device = initialize_gpu()
        settings.device = device
        self.model_manager = ModelManager()
        logger.info(f"DAMO BentoML service initialized on device={device}")

    def _execute(self, task_type: TaskType, request: ExecuteRequest) -> ExecuteResponse:
        result = self.model_manager.infer(
            model_id=request.model_id, task_type=task_type,
            input_data=request.input, params=request.params,
            force_reload=request.force_reload,
        )
        return ExecuteResponse(
            model_id=request.model_id, task_type=task_type.value,
            result=result,
            vram_usage_percent=self.model_manager.vram_manager.get_vram_usage_percent(),
        )

    @bentoml.api(route="/v1/execute/text")
    def execute_text(self, request: ExecuteRequest) -> ExecuteResponse:
        return self._execute(TaskType.TEXT, request)

    @bentoml.api(route="/v1/execute/audio_tts")
    def execute_tts(self, request: ExecuteRequest) -> ExecuteResponse:
        return self._execute(TaskType.AUDIO_TTS, request)

    @bentoml.api(route="/v1/execute/audio_stt")
    def execute_stt(self, request: ExecuteRequest) -> ExecuteResponse:
        return self._execute(TaskType.AUDIO_STT, request)

    @bentoml.api(route="/v1/execute/image")
    def execute_image(self, request: ExecuteRequest) -> ExecuteResponse:
        return self._execute(TaskType.IMAGE, request)

    @bentoml.api(route="/v1/models/status")
    def models_status(self) -> ModelStatusResponse:
        statuses = self.model_manager.get_all_model_status()
        items = [ModelStatusItem(**s) for s in statuses]
        return ModelStatusResponse(
            models=items,
            vram_usage_percent=self.model_manager.vram_manager.get_vram_usage_percent(),
            active_downloads=list(self.model_manager.download_manager.active_downloads),
        )

    @bentoml.api(route="/v1/models/purge")
    def purge_models(self) -> PurgeResponse:
        self.model_manager.purge_all()
        return PurgeResponse(message="All models purged from VRAM")

    @bentoml.api(route="/v1/models/fetch")
    def fetch_model(self, request: FetchRequest) -> FetchResponse:
        path = self.model_manager.fetch_model(request.model_id)
        return FetchResponse(model_id=request.model_id, path=path,
                             message=f"Model {request.model_id} is available on disk")

    @bentoml.api(route="/health")
    def health(self) -> dict:
        return {"status": "ok", "device": settings.device,
                "vram_usage_percent": self.model_manager.vram_manager.get_vram_usage_percent()}
