FROM vllm/vllm-openai:nightly-74fe80ee9594bbc6c0d0c979dbb9d56fae0e789b

# Upgrade transformers to main branch (required for qwen3_5_moe support)
RUN uv pip install --system "transformers @ git+https://github.com/huggingface/transformers.git@main"

# Install handler dependencies (vLLM is already in the base image)
RUN uv pip install --system runpod requests huggingface_hub

COPY handler.py /app/handler.py

WORKDIR /app

ENTRYPOINT []
CMD ["python3", "handler.py"]