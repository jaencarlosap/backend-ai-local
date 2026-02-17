from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference.base import BaseInferenceEngine
from src.utils.gpu_utils import clear_gpu_cache
from src.utils.logger import logger


class LLMEngine(BaseInferenceEngine):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self._model = None
        self._tokenizer = None

    def load(self, model_path: str) -> None:
        logger.info(f"Loading LLM: {self.model_id} from {model_path}")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cuda" and self._model.device.type != "cuda":
            self._model = self._model.to(self.device)
        self._loaded = True
        logger.info(f"LLM loaded: {self.model_id}")

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False
        clear_gpu_cache()
        logger.info(f"LLM unloaded: {self.model_id}")

    def infer(self, input_data: Any, params: Dict[str, Any]) -> Any:
        prompt = str(input_data)
        max_length = params.get("max_length", 100)
        temperature = params.get("temperature", 1.0)
        top_p = params.get("top_p", 1.0)
        do_sample = params.get("do_sample", temperature != 1.0)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": text}

    def get_vram_usage_mb(self) -> float:
        if self._model is None:
            return 0.0
        return sum(
            p.nelement() * p.element_size() for p in self._model.parameters()
        ) / (1024 ** 2)
