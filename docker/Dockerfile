# syntax=docker/dockerfile:1

# --------------- builder stage ---------------
FROM python:3.10-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install heavy/stable deps first (torch, transformers, diffusers, TTS).
# This layer is cached as long as requirements-heavy.txt doesn't change.
COPY requirements-heavy.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements-heavy.txt

# Install lightweight/frequently-changing deps separately.
COPY requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# --------------- runtime stage ---------------
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY src/ ./src/

RUN mkdir -p /app/models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
