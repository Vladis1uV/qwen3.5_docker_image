FROM vllm/vllm-openai:v0.8.5

RUN /opt/venv/bin/pip install runpod requests huggingface_hub

COPY handler.py /app/handler.py

WORKDIR /app

ENTRYPOINT []
CMD ["/opt/venv/bin/python3", "handler.py"]