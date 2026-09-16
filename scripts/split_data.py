"""v1 synthetic split of examples.jsonl: eval = held-out docs; val = a random 10% of the remaining rows (same docs as train)."""
import argparse
import json
import random
from pathlib import Path

import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.DATA / "examples.jsonl"))
    ap.add_argument("--eval-docs", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    docs = sorted({r["doc"] for r in rows})
    rng = random.Random(args.seed)
    eval_docs = set(rng.sample(docs, min(args.eval_docs, len(docs) // 3)))

    eval_rows = [r for r in rows if r["doc"] in eval_docs]
    rng.shuffle(eval_rows)  # so --limit N evals sample every doc type
    rest = [r for r in rows if r["doc"] not in eval_docs]
    rng.shuffle(rest)
    n_val = max(1, int(len(rest) * args.val_frac))
    val_rows, train_rows = rest[:n_val], rest[n_val:]

    for name, part in [("train", train_rows), ("val", val_rows), ("eval", eval_rows)]:
        p = common.DATA / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in part))
        print(f"{name}: {len(part)} rows ({len({r['doc'] for r in part})} docs) -> {p}")
    print(f"eval docs: {sorted(eval_docs)}")


if __name__ == "__main__":
    main()
