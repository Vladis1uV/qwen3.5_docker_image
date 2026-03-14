FROM vllm/vllm-openai:v0.8.5

# Upgrade transformers to support Qwen3.5 architecture
RUN uv pip install --system --upgrade transformers

RUN uv pip install --system --upgrade vllm

RUN uv pip install --system runpod requests huggingface_hub && \
    python -c "import runpod; print('runpod OK')" || echo "runpod MISSING"

COPY handler.py /app/handler.py

WORKDIR /app

ENTRYPOINT []
CMD ["uv", "run", "handler.py"]