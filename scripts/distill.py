"""Compile-verified teacher distillation: sample K fixes per row from a teacher, score each with
run_eval.score_response (compile + strict PDF match when a ref exists), keep the best verified one
as an SFT target. Rows are written incrementally (resumable); failures are recorded with ok=false.

  distill.py --data ../data/texse_train.jsonl --teacher gateway:anthropic/claude-fable-5.1 --k 2
"""
import argparse
import asyncio
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common
from run_eval import sample_gateway, sample_tinker, score_response


def best(scored: list[tuple[dict, dict]]) -> tuple[dict, dict] | None:
    ok = [(s, r) for s, r in scored if r["compiled"]]
    if not ok:
        return None
    return min(ok, key=lambda sr: (not sr[1]["strict"], sr[1]["diff_lines"] or 0, len(sr[0]["response"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher", required=True, help="gateway:<slug> | tinker://... | inkling-small")
    ap.add_argument("--k", type=int, default=2, help="samples per row (first at T=0, rest at T=0.8)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    common.load_env()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    slug = args.teacher.split(":")[-1].split("/")[-1].replace(".", "-")
    out = Path(args.out) if args.out else common.DATA / f"distill_{Path(args.data).stem}_{slug}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    refs = {}
    for name in ("refs.json", "refs_pool.json", "refs_skew.json"):  # same merge as run_eval
        if (common.DATA / name).exists():
            refs.update(json.loads((common.DATA / name).read_text()))
    print(f"{len(rows)} rows, {len(done)} done, {len(todo)} to do, teacher={args.teacher}, k={args.k} -> {out}", flush=True)

    gateway = args.teacher.startswith("gateway:")
    model = args.teacher.split(":", 1)[1] if gateway else (
        "thinkingmachines/Inkling-Small" if args.teacher == "inkling-small" else args.teacher)
    n_ok = n_strict = 0
    t0 = time.monotonic()
    with out.open("a") as fh, ProcessPoolExecutor(6) as ex:
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            samples: list[list[dict]] = [[] for _ in chunk]
            for j in range(args.k if gateway else 1):
                temp = 0.0 if j == 0 else 0.8
                if gateway:
                    got = asyncio.run(sample_gateway(chunk, model, args.concurrency, temperature=temp))
                else:
                    got = sample_tinker(chunk, model)
                for i, s in enumerate(got):
                    samples[i].append(s)
            futs = [[ex.submit(score_response, row, s["response"], refs.get(row["doc"])) for s in ss]
                    for row, ss in zip(chunk, samples)]
            for row, ss, fs in zip(chunk, samples, futs):
                scored = [(s, f.result()) for s, f in zip(ss, fs)]
                pick = best(scored)
                rec = {"id": row["id"], "doc": row["doc"], "mutation": row["mutation"], "lane": row.get("lane"),
                       "teacher": args.teacher, "prompt": row["prompt"], "ok": pick is not None}
                if pick:
                    s, r = pick
                    rec.update(target=s["response"].strip(), strict=r["strict"], diff_lines=r["diff_lines"],
                               out_tokens=s["out_tokens"])
                    n_ok += 1
                    n_strict += r["strict"]
                else:
                    rec["errors"] = [(r.get("apply_error") or "compile failed")[:80] for _, r in scored]
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n = c + len(chunk)
            el = time.monotonic() - t0
            print(f"[{n}/{len(todo)}] ok={n_ok} ({n_ok / n:.0%}) strict={n_strict}  {el / 60:.1f}min, "
                  f"ETA {el / n * (len(todo) - n) / 60:.0f}min", flush=True)
    print(f"done: {n_ok}/{len(todo)} verified targets ({n_strict} strict) -> {out}")


if __name__ == "__main__":
    main()
