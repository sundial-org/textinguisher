"""Assemble an SFT mix from data/<file>:<weight> specs (weight 2 = twice, 0.5 = half). Distilled rows
(distill.py output) are kept only if verified (ok) and not deletion-shaped (fix_env.deleted_chars <= 60,
diff_lines <= 30). No held-out filtering here: every input file must already be a train split. The val file
(--val) is copied alongside as <out with train->val>.

  build_round5.py --mix texse_train.jsonl:2,distill_texse_train_claude-fable-5-1.jsonl:2,project_train_all.jsonl:1,\
train_hard2.jsonl:1,train.jsonl:0.25 --out train5.jsonl
"""
import argparse
import json
import random
from collections import Counter

import common
from fix_env import deleted_chars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", default="train5.jsonl")
    ap.add_argument("--val", default="val3.jsonl", help="val file to copy alongside (val5.jsonl)")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--strict-only", action="store_true", help="teacher rows: keep strict ones (pool rows: minimal edits only)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows, src = [], Counter()
    for spec in args.mix.split(","):
        name, _, w = spec.partition(":")
        w = float(w or 1)
        part = [json.loads(l) for l in (common.DATA / name).read_text().splitlines() if l.strip()]
        if part and part[0].get("teacher"):
            part = [r for r in part if r.get("ok") and deleted_chars(r["target"]) <= 60 and (r.get("diff_lines") or 0) <= 30]
            if args.strict_only:
                part = [r for r in part if r.get("strict") or ("pool" in r["doc"] and (r.get("diff_lines") or 0) <= 10 and deleted_chars(r["target"]) <= 20)]
        part = [{"prompt": r["prompt"], "target": r["target"], "id": r["id"], "src": name} for r in part]
        keep = part * int(w) + [r for r in part if rng.random() < w - int(w)]
        src[name] = len(keep)
        rows += keep
    rng.shuffle(rows)
    (common.DATA / args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    val = (common.DATA / args.val).read_text()
    (common.DATA / args.out.replace("train", "val")).write_text(val)
    print(f"{args.out}: {len(rows)} rows {dict(src)}; val copied from {args.val}")


if __name__ == "__main__":
    main()
