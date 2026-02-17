import base64
import io
from typing import Any, Dict

import torch

from src.inference.base import BaseInferenceEngine
from src.utils.gpu_utils import clear_gpu_cache
from src.utils.logger import logger


class ImageEngine(BaseInferenceEngine):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self._pipeline = None

    def load(self, model_path: str) -> None:
        from diffusers import StableDiffusionPipeline

        logger.info(f"Loading image model: {self.model_id}")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._pipeline = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        )
        if self.device == "cuda":
            self._pipeline = self._pipeline.to(self.device)
        self._loaded = True
        logger.info(f"Image model loaded: {self.model_id}")

    def unload(self) -> None:
        del self._pipeline
        self._pipeline = None
        self._loaded = False
        clear_gpu_cache()
        logger.info(f"Image model unloaded: {self.model_id}")

    def infer(self, input_data: Any, params: Dict[str, Any]) -> Any:
        prompt = str(input_data)
        guidance_scale = params.get("guidance_scale", 7.5)
        num_inference_steps = params.get("num_inference_steps", 50)
        width = params.get("width", 512)
        height = params.get("height", 512)

        with torch.no_grad():
            result = self._pipeline(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
            )

        image = result.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image_base64": image_b64, "format": "png"}

    def get_vram_usage_mb(self) -> float:
        if self._pipeline is None:
            return 0.0
        total = 0
        for component_name in ["unet", "vae", "text_encoder"]:
            component = getattr(self._pipeline, component_name, None)
            if component and hasattr(component, "parameters"):
                total += sum(
                    p.nelement() * p.element_size() for p in component.parameters()
                )
        return total / (1024 ** 2)
