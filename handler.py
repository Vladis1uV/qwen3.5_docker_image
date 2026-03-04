import runpod
import requests
import subprocess
import time
import os
import json

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-35B-A3B-FP8")
TENSOR_PARALLEL = os.environ.get("TENSOR_PARALLEL_SIZE", "8")   # GPUs per node
MAX_MODEL_LEN   = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPU_UTIL        = os.environ.get("GPU_MEMORY_UTILIZATION", "0.95")
HOST            = "0.0.0.0"
PORT            = 8000
VLLM_URL        = f"http://localhost:{PORT}"

# ── Launch vLLM server ────────────────────────────────────────────────────────
def start_vllm():
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  "Qwen/Qwen3.5-35B-A3B-FP8",
        "--tensor-parallel-size",   TENSOR_PARALLEL,
        "--max-model-len",          "32768",   # reduce from 262144 to save VRAM
        "--gpu-memory-utilization", GPU_UTIL,
        "--reasoning-parser",       "qwen3",   # ← required for this model
        "--dtype",                  "auto",
        "--disable-log-requests",
    ]

    # Optional: load from local cache if pre-baked into image or network volume
    hf_cache = os.environ.get("HF_HOME")
    if hf_cache:
        cmd += ["--download-dir", hf_cache]

    hf_token = os.environ.get("HF_TOKEN")
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token

    subprocess.Popen(cmd, env=env)

    # Wait until server is healthy
    print("Waiting for vLLM server to be ready...")
    for _ in range(600):          # up to 10 min for large model download
        try:
            r = requests.get(f"{VLLM_URL}/health", timeout=2)
            if r.status_code == 200:
                print("vLLM server is ready.")
                return
        except Exception:
            pass
        time.sleep(1)

    raise RuntimeError("vLLM server failed to start within timeout")


# ── RunPod handler ────────────────────────────────────────────────────────────
def handler(job):
    job_input = job["input"]

    # Support raw /v1/chat/completions payload
    messages  = job_input.get("messages", [])
    model     = job_input.get("model", MODEL_ID)
    stream    = job_input.get("stream", False)

    # Build request body — pass through all OpenAI-compatible params
    body = {
        "model":       model,
        "messages":    messages,
        "stream":      stream,
        **{k: v for k, v in job_input.items()
           if k not in ("messages", "model", "stream")},
    }

    try:
        resp = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=body,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        return {"error": str(e), "detail": resp.text}
    except Exception as e:
        return {"error": str(e)}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start_vllm()
    runpod.serverless.start({"handler": handler})