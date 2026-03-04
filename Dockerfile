FROM vllm/vllm-openai:latest

# Install the RunPod serverless worker SDK
RUN pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

# Copy the handler
COPY handler.py /app/handler.py

WORKDIR /app

CMD ["python", "handler.py"]