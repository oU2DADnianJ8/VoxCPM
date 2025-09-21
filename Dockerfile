# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY README.md pyproject.toml ./
COPY src ./src
COPY assets ./assets
COPY examples ./examples

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir .

EXPOSE 8000

ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models/transformers \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

VOLUME ["/models"]

CMD ["python3", "-m", "voxcpm.server.app", "--host", "0.0.0.0", "--port", "8000"]
