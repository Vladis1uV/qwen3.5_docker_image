FROM vllm/vllm-openai:v0.8.5

RUN uv pip install --system runpod requests huggingface_hub && \
    python -c "import runpod; print('runpod OK')" || echo "runpod MISSING"

COPY handler.py /app/handler.py

WORKDIR /app

ENTRYPOINT []
CMD ["uv", "run", "handler.py"]