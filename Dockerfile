FROM vllm/vllm-openai:latest

RUN pip install runpod huggingface_hub

COPY handler.py /app/handler.py

WORKDIR /app

CMD ["python", "handler.py"]