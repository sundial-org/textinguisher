"""Modal app: vLLM serving Inkling-Small-NVFP4 (8xH200, TP8: LoRA forces bf16 experts) + latexfix LoRA.

Usage:
  modal run modal_vllm.py::download_weights   # one-time, CPU container -> Volume
  modal deploy modal_vllm.py                  # start server, prints URL
  modal app stop latexfix-vllm                # tear down
Adapter is uploaded separately: modal volume put inkling-small-weights adapter/ /adapter
"""
import subprocess

import modal

MODEL_REPO = "thinkingmachines/Inkling-Small-NVFP4"
MODEL_DIR = "/weights/Inkling-Small-NVFP4"
ADAPTER_DIR = "/weights/adapter"
PORT = 8000

app = modal.App("latexfix-vllm")
vol = modal.Volume.from_name("inkling-small-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "huggingface_hub[hf_transfer]", "tiktoken", "blobfile")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, volumes={"/weights": vol}, timeout=3600, cpu=8, memory=32768)
def download_weights():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_REPO, local_dir=MODEL_DIR)
    vol.commit()
    print("weights downloaded")


@app.function(
    image=image,
    volumes={"/weights": vol},
    gpu="H200:8",
    timeout=3600,
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=3000)
def serve():
    cmd = [
        "vllm", "serve", MODEL_DIR,
        "--served-model-name", "base",
        "--tokenizer-mode", "inkling",
        "--reasoning-parser", "inkling",
        "--trust-remote-code",
        "--tensor-parallel-size", "8",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.90",
        "--enable-lora",
        "--max-lora-rank", "32",
        "--lora-modules", f"latexfix={ADAPTER_DIR}",
        "--port", str(PORT),
    ]
    env = {"VLLM_USE_V2_MODEL_RUNNER": "1", "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
           "INKLING_MULTIMEM_AR": "0",  # required for Inkling LoRA (vLLM validation)
           "VLLM_USE_FLASHINFER_SAMPLER": "0"}  # slim image has no nvcc for flashinfer JIT
    import os
    subprocess.Popen(cmd, env={**os.environ, **env})
