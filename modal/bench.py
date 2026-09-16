"""Sequential latency bench against the Modal vLLM endpoint.

Usage: python bench.py <base_url> [--model latexfix|base|<gateway slug>] [--n 15] [--out results.json] [--key-env AI_GATEWAY_API_KEY]
Streams chat completions (temp 0, max_tokens 2000), records wall, TTFT, output tokens.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
from common import SYSTEM, load_env  # noqa: E402


def bench_one(client, url, model, prompt):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2000,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.monotonic()
    ttft = None
    text = ""
    usage = None
    with client.stream("POST", url + "/v1/chat/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    text += d.get("content") or ""
    wall = time.monotonic() - t0
    out_toks = usage["completion_tokens"] if usage else None
    return {"wall": wall, "ttft": ttft, "output_tokens": out_toks, "first100": text[:100]}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--model", default="latexfix")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default=None)
    ap.add_argument("--key-env", default=None, help="env var holding a bearer token (OpenAI-compatible APIs)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(PROJ / "data" / "eval.jsonl")][: args.n]
    results = []
    load_env()
    headers = {"Authorization": f"Bearer {os.environ[args.key_env]}"} if args.key_env else {}
    with httpx.Client(headers=headers) as client:
        for i, row in enumerate(rows):
            res = bench_one(client, args.base_url.rstrip("/"), args.model, row["prompt"])
            res["id"] = row["id"]
            results.append(res)
            print(f"[{i+1}/{len(rows)}] wall={res['wall']:.2f}s ttft={res['ttft'] or 0:.2f}s "
                  f"toks={res['output_tokens']} | {res['first100'][:80]!r}", flush=True)

    walls = [r["wall"] for r in results]
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]  # None = empty response
    decode_tps = [r["output_tokens"] / (r["wall"] - r["ttft"]) for r in results
                  if r["output_tokens"] and r["ttft"] and r["wall"] > r["ttft"]]
    summary = {
        "model": args.model, "n": len(results),
        "wall_p50": pct(walls, 50), "wall_p90": pct(walls, 90),
        "ttft_p50": pct(ttfts, 50),
        "decode_tps_median": statistics.median(decode_tps) if decode_tps else None,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
