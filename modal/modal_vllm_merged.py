"""Modal app: vLLM serving the LoRA-merged BF16 Inkling-Small (8xH200 TP=8, no LoRA kernels).

  modal deploy modal_vllm_merged.py        # start server, prints URL
  modal app stop latexfix-vllm-merged      # tear down
Merged weights are produced by merge_inkling.py into the same volume.
"""
import os
import subprocess

import modal

MODEL_DIR = "/weights/Inkling-Small-merged-rlM6c_000120"
PORT = 8000

app = modal.App("latexfix-vllm-merged")
vol = modal.Volume.from_name("inkling-small-weights")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "huggingface_hub[hf_transfer]", "tiktoken", "blobfile")
)


@app.function(image=image, volumes={"/weights": vol}, gpu="H200:8", timeout=3600,
              scaledown_window=300, max_containers=1)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=3000)
def serve():
    cmd = [
        "vllm", "serve", MODEL_DIR,
        "--served-model-name", "merged",
        "--tokenizer-mode", "inkling",
        "--reasoning-parser", "inkling",
        "--trust-remote-code",
        "--tensor-parallel-size", "8",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.90",
        "--port", str(PORT),
    ]
    env = {"VLLM_USE_V2_MODEL_RUNNER": "1", "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
           "INKLING_MULTIMEM_AR": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    subprocess.Popen(cmd, env={**os.environ, **env})
