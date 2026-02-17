import torch

from src.utils.logger import logger

_nvml_initialized = False


def _init_nvml():
    global _nvml_initialized
    if _nvml_initialized:
        return True
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_initialized = True
        return True
    except Exception as e:
        logger.warning(f"pynvml init failed (no NVIDIA GPU?): {e}")
        return False


def initialize_gpu() -> str:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logger.info(f"GPU detected: {device_name} ({total_mem:.1f} GB)")
        _init_nvml()
        return "cuda"
    logger.warning("No CUDA GPU available, falling back to CPU")
    return "cpu"


def get_vram_usage_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(0) / (1024 ** 3)


def get_vram_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)


def get_vram_percent() -> float:
    total = get_vram_total_gb()
    if total == 0:
        return 0.0
    return get_vram_usage_gb() / total


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU cache cleared")
