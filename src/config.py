from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hf_token: str = ""
    device: str = "cuda"
    max_vram_gb: float = 8.0
    vram_threshold: float = 0.9
    model_cache_dir: str = "/app/models"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
