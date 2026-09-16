"""Throughput of the deployed vLLM endpoint: N concurrent streams over the eval prompts -> fixes/hour and tokens/s.
  python modal/bench_throughput.py https://cosci--latexfix-vllm-merged-serve.modal.run --concurrency 48 --n 240"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
import time

import httpx

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import common  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("--concurrency", type=int, default=48); ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--out", default="bench_throughput.json")
    a = ap.parse_args()
    if a.n <= 0:
        ap.error("--n must be positive")
    rows = [json.loads(l) for l in open(PROJ / "data" / "texse2_heldout.jsonl")][: a.n]
    sem, done, toks = asyncio.Semaphore(a.concurrency), [], []

    async def one(client, row):
        async with sem:
            t0 = time.monotonic()
            r = await client.post(a.url + "/v1/chat/completions", json={"model": "merged", "temperature": 0, "max_tokens": 600,
                 "messages": [{"role": "system", "content": common.SYSTEM}, {"role": "user", "content": row["prompt"]}]}, timeout=600)
            j = r.json(); done.append(time.monotonic() - t0); toks.append(j["usage"]["completion_tokens"])

    async with httpx.AsyncClient() as client:
        await one(client, rows[0])  # warm
        t0 = time.monotonic()
        await asyncio.gather(*[one(client, r) for r in rows])
        wall = time.monotonic() - t0
    res = {"n": len(rows), "concurrency": a.concurrency, "wall_s": wall, "fixes_per_hour": 3600 * len(rows) / wall,
           "out_tokens_per_s": sum(toks[1:]) / wall, "mean_out_tokens": sum(toks[1:]) / len(rows), "p50_latency_s": sorted(done[1:])[len(rows) // 2]}
    json.dump(res, open(a.out, "w"), indent=1); print(res)


asyncio.run(main())
