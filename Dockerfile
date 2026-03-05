FROM vllm/vllm-openai:v0.8.5

RUN pip install runpod huggingface_hub

COPY handler.py /app/handler.py

WORKDIR /app

ENTRYPOINT []
CMD ["python", "handler.py"]