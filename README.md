# DAMO — Dynamic AI Model Orchestrator

Unified REST API for running multimodal AI inference (Text, Image, Audio, Video) on consumer-grade GPUs. Models are downloaded on-demand from HuggingFace and managed in VRAM with automatic LRU eviction.

## Features

- **On-demand model loading** — Models download from HuggingFace on first request and are cached to disk.
- **VRAM management** — LRU eviction at a configurable threshold (default 90%) keeps your GPU from running out of memory.
- **Multimodal** — Text generation, image generation, text-to-speech, speech-to-text, and video (planned).
- **Swagger docs** — Interactive API documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc).

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (>= 20.10)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)
- An NVIDIA GPU with CUDA-compatible drivers

## Quick Start

### 1. Clone and configure

```bash
git clone <your-repo-url> backend-ai-local
cd backend-ai-local
cp .env.example .env
```

Edit `.env` to set your values:

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace access token (required for gated models) | _(empty)_ |
| `DEVICE` | Compute device (`cuda` or `cpu`) | `cuda` |
| `MAX_VRAM_GB` | Maximum VRAM budget in GB (used as fallback when GPU detection fails) | `8.0` |
| `VRAM_THRESHOLD` | VRAM usage threshold before LRU eviction triggers (0.0–1.0) | `0.9` |
| `MODEL_CACHE_DIR` | Directory for downloaded models inside the container | `/app/models` |
| `LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

### 2. Run with Docker Compose (recommended)

```bash
docker compose -f docker/docker-compose.yml up --build
```

This will:
- Build the Docker image with CUDA 12.1 + Python 3.10
- Start the API server on port **8000**
- Mount `./models` and `./cache` as volumes so downloaded models persist across restarts
- Reserve 1 NVIDIA GPU for the container

To run in detached mode:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

To stop:

```bash
docker compose -f docker/docker-compose.yml down
```

To view logs:

```bash
docker compose -f docker/docker-compose.yml logs -f
```

### 3. Run with Docker directly

If you prefer running without Compose:

**Build the image:**

```bash
docker build -t damo -f docker/Dockerfile .
```

**Run the container:**

```bash
docker run -d \
  --name damo \
  --gpus 1 \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/root/.cache/huggingface \
  --env-file .env \
  damo
```

**Stop and remove:**

```bash
docker stop damo && docker rm damo
```

## Verify It's Running

```bash
# Health check
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "device": "cuda",
  "vram_usage_percent": 0.0
}
```

## API Usage

### Generate text

```bash
curl -X POST http://localhost:8000/v1/execute/text \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt2",
    "input": "The future of AI is",
    "params": {"max_length": 50, "temperature": 0.7}
  }'
```

### Generate an image

```bash
curl -X POST http://localhost:8000/v1/execute/image \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "stabilityai/stable-diffusion-2-1",
    "input": "a photo of an astronaut riding a horse on mars",
    "params": {"num_inference_steps": 30, "width": 512, "height": 512}
  }'
```

### Pre-download a model

```bash
curl -X POST http://localhost:8000/v1/models/fetch \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt2"}'
```

### Check model status

```bash
curl http://localhost:8000/v1/models/status
```

### Free GPU memory

```bash
curl -X DELETE http://localhost:8000/v1/models/purge
```

## API Documentation

Once the server is running, open your browser:

| URL | Description |
|---|---|
| [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger UI — interactive API explorer |
| [http://localhost:8000/redoc](http://localhost:8000/redoc) | ReDoc — alternative API reference |
| [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json) | Raw OpenAPI 3.1 JSON schema |

## Project Structure

```
backend-ai-local/
├── src/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Environment settings
│   ├── api/v1/
│   │   ├── execute.py           # POST /v1/execute/{task_type}
│   │   ├── models.py            # Model management endpoints
│   │   └── schemas.py           # Request/response schemas
│   ├── core/
│   │   ├── model_manager.py     # Central orchestrator
│   │   ├── vram_manager.py      # GPU memory tracking & LRU eviction
│   │   ├── download_manager.py  # Async HuggingFace downloads
│   │   └── task_router.py       # Task type → engine mapping
│   ├── inference/
│   │   ├── base.py              # Abstract engine interface
│   │   ├── llm_engine.py        # Text generation
│   │   ├── image_engine.py      # Image generation
│   │   ├── tts_engine.py        # Text-to-Speech
│   │   ├── stt_engine.py        # Speech-to-Text
│   │   └── video_engine.py      # Video (placeholder)
│   ├── models/
│   │   ├── enums.py             # TaskType, ModelState
│   │   └── model_info.py        # Model metadata dataclass
│   └── utils/
│       ├── gpu_utils.py         # CUDA helpers
│       ├── exceptions.py        # Custom exceptions
│       └── logger.py            # Logging setup
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── .gitignore
```

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Local Development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## License

MIT
