FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Link CUDA compat libraries so vLLM can find libcudart.so.12
RUN ldconfig /usr/local/cuda/compat/

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