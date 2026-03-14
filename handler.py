import runpod
import requests
import subprocess
import time
import os
import json
import sys
import threading

print("=== [1/6] IMPORTS OK ===", flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-35B-A3B-FP8")
TENSOR_PARALLEL = os.environ.get("TENSOR_PARALLEL_SIZE", "1")
MAX_MODEL_LEN   = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPU_UTIL        = os.environ.get("GPU_MEMORY_UTILIZATION", "0.92")
HF_HOME         = os.environ.get("HF_HOME", "/runpod-volume/.cache/huggingface")
PORT            = 8000
VLLM_URL        = f"http://localhost:{PORT}"

print("=== [2/6] CONFIG LOADED ===", flush=True)
print(f"  MODEL_ID:              {MODEL_ID}", flush=True)
print(f"  TENSOR_PARALLEL_SIZE:  {TENSOR_PARALLEL}", flush=True)
print(f"  MAX_MODEL_LEN:         {MAX_MODEL_LEN}", flush=True)
print(f"  GPU_MEMORY_UTILIZATION:{GPU_UTIL}", flush=True)
print(f"  HF_HOME:               {HF_HOME}", flush=True)

# ── Check volume / model files ────────────────────────────────────────────────
print("=== [3/6] CHECKING VOLUME & MODEL FILES ===", flush=True)
if os.path.exists("/runpod-volume"):
    print(f"  /runpod-volume EXISTS: {os.listdir('/runpod-volume')}", flush=True)
else:
    print("  WARNING: /runpod-volume does NOT exist — no network volume mounted!", flush=True)

if HF_HOME and os.path.exists(HF_HOME):
    print(f"  HF_HOME exists: {os.listdir(HF_HOME)}", flush=True)
else:
    print(f"  WARNING: HF_HOME path not found: {HF_HOME}", flush=True)

# ── Check GPU ────────────────────────────────────────────────────────────────
print("=== [4/6] CHECKING GPU ===", flush=True)
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    print(f"  GPU info: {result.stdout.strip()}", flush=True)
except Exception as e:
    print(f"  WARNING: nvidia-smi failed: {e}", flush=True)

# ── Launch vLLM server ────────────────────────────────────────────────────────
def stream_logs(process):
    """Stream vLLM logs in a background thread."""
    for line in iter(process.stdout.readline, ""):
        print(f"  [vLLM] {line.strip()}", flush=True)

def start_vllm():
    print("=== [5/6] STARTING VLLM SERVER ===", flush=True)

    cmd = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--tensor-parallel-size", TENSOR_PARALLEL,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", GPU_UTIL,
        "--trust-remote-code",
        "--max-num-seqs", "16",
        "--dtype", "auto",
        "--disable-log-requests",
        "--download-dir", HF_HOME,
        "--enable-chunked-prefill",
        "--reasoning-parser", "qwen3",  # Required per Qwen3.5 model card
    ]

    print(f"  vLLM command: {' '.join(cmd)}", flush=True)

    process = subprocess.Popen(
        cmd,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream logs in background so readline() doesn't block health checks
    log_thread = threading.Thread(target=stream_logs, args=(process,), daemon=True)
    log_thread.start()

    print("  Waiting for vLLM to be ready (up to 600s)...", flush=True)
    for i in range(600):
        if process.poll() is not None:
            print(f"  ERROR: vLLM exited with code {process.poll()}", flush=True)
            sys.exit(1)

        try:
            r = requests.get(f"{VLLM_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"  vLLM is READY after {i}s!", flush=True)
                return
        except Exception:
            pass

        time.sleep(1)

    print("  ERROR: vLLM did not become ready within 600 seconds", flush=True)
    sys.exit(1)


# ── RunPod handler ────────────────────────────────────────────────────────────
def handler(job):
    print(f"=== REQUEST: job={job.get('id')} ===", flush=True)

    job_input = job["input"]
    messages  = job_input.get("messages", [])
    model     = job_input.get("model", MODEL_ID)
    stream    = job_input.get("stream", False)

    body = {
        "model":    model,
        "messages": messages,
        "stream":   stream,
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
    try:
        print("=== WORKER STARTING ===", flush=True)
        start_vllm()
        print("=== VLLM READY — STARTING RUNPOD HANDLER ===", flush=True)
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"=== FATAL ERROR: {e} ===", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)