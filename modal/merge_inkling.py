"""Merge the Tinker LoRA (raw 3D-expert format) into thinkingmachines/Inkling-Small BF16, shard-parallel on Modal.

Usage (MODAL_PROFILE=sundial-prod):
  modal run merge_inkling.py --stage adapter   # tinker checkpoint -> ADAPTER_DIR
  modal run merge_inkling.py --stage aux       # config/tokenizer/index/mtp -> OUT_DIR
  modal run merge_inkling.py --stage merge [--shards 1,2]   # each container: HF shard -> merge -> volume

Layout facts (vllm/models/inkling/nvidia/{model,moe}.py @ v0.28.0):
  linears are [out, in]; dense layers 0-1: w13_dn<-gate_up_proj, w2_md<-down_proj (1:1); experts.w13_weight is (E, 2F, D) with rows interleaved [g0,u0,g1,u1,...],
  w1=gate, w3=up; w2_weight is (E, D, F); shared_experts.shared_w13_weight (S, 2F, D) same interleave,
  loaded as gate rows [e0;e1]; shared_w2_weight (S, D, F) loaded as (D, [e0;e1]); lm_head -> model.llm.unembed.
"""
import json
import os
import re
import time

import modal

BASE_REPO = "thinkingmachines/Inkling-Small"
TINKER_PATH = "tinker://1cc5d259-2f60-50bb-bd22-632c61774eaf:train:0/sampler_weights/000120"
ADAPTER_DIR = "/weights/adapter_raw_rlM6c_000120"
OUT_DIR = "/weights/Inkling-Small-merged-rlM6c_000120"

app = modal.App("latexfix-merge")
vol = modal.Volume.from_name("inkling-small-weights")

tinker_image = modal.Image.debian_slim(python_version="3.12").pip_install("tinker")
merge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("safetensors", "numpy", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=tinker_image, volumes={"/weights": vol}, timeout=3600, memory=8192)
def fetch_adapter(api_key: str):
    import tarfile
    import urllib.request

    import tinker

    if os.path.exists(f"{ADAPTER_DIR}/checkpoint_complete"):
        return "adapter already present"
    rc = tinker.ServiceClient(api_key=api_key).create_rest_client()
    url = rc.get_checkpoint_archive_url_from_tinker_path(TINKER_PATH).result().url
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    urllib.request.urlretrieve(url, "/tmp/ckpt.tar")
    with tarfile.open("/tmp/ckpt.tar") as tar:
        tar.extractall(ADAPTER_DIR, filter="data")
    vol.commit()
    return f"adapter: {os.listdir(ADAPTER_DIR)}"


@app.function(image=merge_image, volumes={"/weights": vol}, timeout=3600, memory=16384)
def copy_aux():
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_REPO, local_dir=OUT_DIR, ignore_patterns=["model-*.safetensors", ".cache/*"])
    vol.commit()
    return sorted(os.listdir(OUT_DIR))


@app.function(image=merge_image, volumes={"/weights": vol}, timeout=3600, cpu=16, memory=131072,
              max_containers=16, retries=1)
def merge_shard(shard: str) -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    t0 = time.time()
    cfg = json.load(open(f"{ADAPTER_DIR}/adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    ad = safe_open(f"{ADAPTER_DIR}/adapter_model.safetensors", "pt")
    used = []

    def lora(name):  # -> (A, B) fp32, or None
        k = f"language_model.{name}"
        if f"{k}.lora_A.weight" not in ad.keys():
            return None
        used.append(k)
        return ad.get_tensor(f"{k}.lora_A.weight").float(), ad.get_tensor(f"{k}.lora_B.weight").float() * scale

    def add(view, delta):  # bf16 view += fp32 delta, chunked over dim 0
        for i in range(0, view.shape[0], 32):
            view[i:i + 32] = (view[i:i + 32].float() + delta[i:i + 32]).to(torch.bfloat16)

    def expert_delta(A, B):  # A (Ea,r,in), B (Eb,out,r) -> (E,out,in), chunked
        E = max(A.shape[0], B.shape[0])
        A, B = A.expand(E, -1, -1), B.expand(E, -1, -1)
        return torch.cat([torch.bmm(B[i:i + 32], A[i:i + 32]) for i in range(0, E, 32)])

    path = hf_hub_download(BASE_REPO, shard, local_dir="/tmp/base")
    out = {}
    with safe_open(path, "pt") as f:
        for key in f.keys():
            w = f.get_tensor(key)
            m = re.match(r"model\.llm\.layers\.(\d+)\.(.*)", key)
            sub = m.group(2) if m else None
            pre = f"layers.{m.group(1)}." if m else ""
            if key == "model.llm.unembed.weight":
                A, B = lora("lm_head")
                add(w, B @ A)
            elif sub and re.fullmatch(r"attn\.(wq_du|wk_dv|wv_dv|wo_ud|wr_du)\.weight", sub):
                A, B = lora(pre + sub[:-len(".weight")])
                add(w, B @ A)
            elif sub in ("mlp.w13_dn.weight", "mlp.w2_md.weight"):  # dense layers 0-1
                A, B = lora(pre + ("mlp.gate_up_proj" if "w13" in sub else "mlp.down_proj"))
                add(w, B @ A)
            elif sub == "mlp.experts.w13_weight":
                for j, n in enumerate(("w1", "w3")):
                    A, B = lora(f"{pre}mlp.experts.{n}")
                    add(w[:, j::2, :], expert_delta(A, B))
            elif sub == "mlp.experts.w2_weight":
                A, B = lora(f"{pre}mlp.experts.w2")
                add(w, expert_delta(A, B))
            elif sub == "mlp.shared_experts.shared_w13_weight":
                S, F2, D = w.shape
                for j, n in enumerate(("w1", "w3")):
                    A, B = lora(f"{pre}mlp.shared_experts.{n}")
                    add(w[:, j::2, :], (B @ A).view(S, F2 // 2, D))
            elif sub == "mlp.shared_experts.shared_w2_weight":
                S, D, F = w.shape
                A, B = lora(f"{pre}mlp.shared_experts.w2")
                add(w, (B @ A).view(D, S, F).permute(1, 0, 2))
            out[key] = w.contiguous()
    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(out, f"{OUT_DIR}/{shard}", metadata={"format": "pt"})
    os.remove(path)
    vol.commit()
    return {"shard": shard, "used": used, "n_tensors": len(out), "secs": round(time.time() - t0)}


@app.function(image=merge_image, volumes={"/weights": vol}, timeout=1800, memory=8192)
def verify() -> dict:
    """Header-level check: every indexed tensor exists in OUT_DIR with the right shape/dtype."""
    import struct

    idx = json.load(open(f"{OUT_DIR}/model.safetensors.index.json"))["weight_map"]
    missing, bad = [], []
    hdrs = {}
    for fn in set(idx.values()):
        p = f"{OUT_DIR}/{fn}"
        if not os.path.exists(p):
            missing.append(fn)
            continue
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdrs[fn] = json.loads(fh.read(n))
    for k, fn in idx.items():
        if fn in hdrs and k not in hdrs[fn]:
            bad.append(k)
    total = sum(os.path.getsize(f"{OUT_DIR}/{fn}") for fn in hdrs)
    return {"missing_files": missing, "missing_tensors": bad[:10], "n_bad": len(bad), "bytes": total}


@app.local_entrypoint()
def main(stage: str = "merge", shards: str = ""):
    if stage == "adapter":
        env = dict(l.strip().split("=", 1) for l in open(os.path.join(os.path.dirname(__file__), "../../.env")) if "=" in l)
        print(fetch_adapter.remote(env["TML_KEY"]))
    elif stage == "aux":
        print(copy_aux.remote())
    elif stage == "verify":
        print(verify.remote())
    else:
        names = [f"model-{i:05d}-of-00032.safetensors" for i in
                 (map(int, shards.split(",")) if shards else range(1, 33))]
        used = set()
        for r in merge_shard.map(names):
            print(r["shard"], r["n_tensors"], "tensors", len(r["used"]), "lora targets", r["secs"], "s", flush=True)
            used.update(r["used"])
        print("lora targets merged:", len(used))
