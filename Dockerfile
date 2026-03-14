FROM vllm/vllm-openai:nightly-74fe80ee9594bbc6c0d0c979dbb9d56fae0e789b

RUN pip install "transformers @ git+https://github.com/huggingface/transformers.git@main" --break-system-packages

RUN pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly --break-system-packages

RUN pip install --system runpod requests huggingface_hub && \
    python -c "import runpod; print('runpod OK')" || echo "runpod MISSING"

COPY handler.py /app/handler.py

WORKDIR /app

ENTRYPOINT []
CMD ["python3", "handler.py"]