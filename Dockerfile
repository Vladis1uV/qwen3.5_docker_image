FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Install vLLM stable 0.17.1 (includes Qwen3.5 support)
RUN uv pip install --system vllm --torch-backend=auto

# Install handler dependencies
RUN uv pip install --system runpod requests huggingface_hub

COPY handler.py /app/handler.py

WORKDIR /app

ENTRYPOINT []
CMD ["python3", "handler.py"]