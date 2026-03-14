FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Install vLLM nightly (includes compatible transformers) per Qwen3.5 model card
RUN uv pip install --system vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly

# Install handler dependencies
RUN uv pip install --system runpod requests huggingface_hub

COPY handler.py /app/handler.py

WORKDIR /app

ENTRYPOINT []
CMD ["python3", "handler.py"]