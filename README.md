# DAMO — Dynamic AI Model Orchestrator

Unified REST API for running multimodal AI inference (Text, Image, Audio, Video) on consumer-grade GPUs. Models are downloaded on-demand from HuggingFace and managed in VRAM with automatic LRU eviction. Built on [BentoML](https://docs.bentoml.com/) for GPU-aware containerization and serving.

## Features

- **On-demand model loading** — Models download from HuggingFace on first request and are cached to disk.
- **VRAM management** — LRU eviction at a configurable threshold (default 90%) keeps your GPU from running out of memory.
- **Multimodal** — Text generation, image generation, text-to-speech, speech-to-text, and video (planned).
- **BentoML serving** — Built-in containerization, GPU-aware image building, and production-ready serving infrastructure.

## Prerequisites

- Python 3.10+
- [Docker](https://docs.docker.com/get-docker/) (>= 20.10) — for containerized deployment
- [Docker Compose](https://docs.docker.com/compose/install/) (v2) — for containerized deployment
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

Build the BentoML image first, then deploy with Compose:

```bash
# Build the Bento and containerize it
bentoml build
bentoml containerize damo:latest --image-tag damo:latest

# Start with Docker Compose
docker compose -f docker/docker-compose.yml up -d
```

This will:
- Use the BentoML-built Docker image with CUDA 11.8 + Python 3.10
- Start the API server on port **3000**
- Mount `./models` and `./cache` as volumes so downloaded models persist across restarts
- Reserve 1 NVIDIA GPU for the container

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

```bash
# Build the Bento and containerize
bentoml build
bentoml containerize damo:latest --image-tag damo:latest

# Run the container
docker run -d \
  --name damo \
  --gpus 1 \
  -p 3000:3000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/root/.cache/huggingface \
  --env-file .env \
  damo:latest
```

**Stop and remove:**

```bash
docker stop damo && docker rm damo
```

## Verify It's Running

```bash
# Health check
curl http://localhost:3000/health
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
curl -X POST http://localhost:3000/v1/execute/text \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt2",
    "input": "The future of AI is",
    "params": {"max_length": 50, "temperature": 0.7}
  }'
```

### Generate an image

```bash
curl -X POST http://localhost:3000/v1/execute/image \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "stabilityai/stable-diffusion-2-1",
    "input": "a photo of an astronaut riding a horse on mars",
    "params": {"num_inference_steps": 30, "width": 512, "height": 512}
  }'
```

### Text-to-Speech

```bash
curl -X POST http://localhost:3000/v1/execute/audio_tts \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "Qwen/Qwen3-TTS",
    "input": "Hello, how are you?",
    "params": {"speaker": "Chelsie", "speed": 1.0}
  }'
```

### Speech-to-Text

```bash
curl -X POST http://localhost:3000/v1/execute/audio_stt \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "openai/whisper-small",
    "input": "<base64-encoded-audio>",
    "params": {"language": "en", "task": "transcribe"}
  }'
```

### Pre-download a model

```bash
curl -X POST http://localhost:3000/v1/models/fetch \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt2"}'
```

### Check model status

```bash
curl http://localhost:3000/v1/models/status
```

### Free GPU memory

```bash
curl -X POST http://localhost:3000/v1/models/purge
```

## Project Structure

```
backend-ai-local/
├── service.py                   # BentoML service entry point
├── bentofile.yaml               # BentoML build configuration
├── src/
│   ├── config.py                # Environment settings
│   ├── schemas.py               # Request/response Pydantic models
│   ├── core/
│   │   ├── model_manager.py     # Central orchestrator
│   │   ├── vram_manager.py      # GPU memory tracking & LRU eviction
│   │   ├── download_manager.py  # HuggingFace downloads
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
bentoml serve service:DAMOService --reload
```

The server will start on `http://localhost:3000`.

## License

MIT
