"""Build a local skeleton of thinkingmachines/Inkling-Small (BF16) for build_lora_adapter.

build_lora_adapter only reads config.json + safetensors HEADERS (keys/shapes),
so we fetch each shard's header via HTTP Range and create sparse files of the
correct total size instead of downloading ~550GB.
"""
import json
import os
import struct
import sys

import httpx

REPO = "thinkingmachines/Inkling-Small"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_skeleton")
os.makedirs(OUT, exist_ok=True)

client = httpx.Client(follow_redirects=True, timeout=120)
info = client.get(f"https://huggingface.co/api/models/{REPO}").json()
files = [s["rfilename"] for s in info["siblings"]]

for f in files:
    url = f"https://huggingface.co/{REPO}/resolve/main/{f}"
    dst = os.path.join(OUT, f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if f.endswith(".safetensors"):
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        r = client.get(url, headers={"Range": "bytes=0-7"})
        header_len = struct.unpack("<Q", r.content)[0]
        total = int(r.headers["content-range"].split("/")[1])
        h = client.get(url, headers={"Range": f"bytes=0-{7 + header_len}"})
        with open(dst, "wb") as fh:
            fh.write(h.content)
            fh.truncate(total)  # sparse: reads as zeros, no disk used
        print(f, "header", header_len, "total", total)
    else:
        if os.path.exists(dst):
            continue
        r = client.get(url)
        if r.status_code == 200 and len(r.content) < 50_000_000:
            open(dst, "wb").write(r.content)
            print(f, len(r.content))
        else:
            print("skip", f, r.status_code, file=sys.stderr)
print("skeleton done:", OUT)
