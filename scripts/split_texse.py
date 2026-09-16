"""By-question train/eval split of data/texse_eval.jsonl, stratified by error_category
(~25% eval; every category with >=2 rows appears in both halves; singletons go to train).
Rows already probed (ids in --probed result files, skipped when absent) are forced into eval, so the held-out half
over-represents the rows probed before the split.

  python3 split_texse.py [--probed ../out/texse_gemini.json] [--frac 0.25]
  python3 split_texse.py --data ../data/texse2_labeled.jsonl --out ../data/texse2_split.json --probed --jsonl texse2
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.DATA / "texse_eval.jsonl"))
    ap.add_argument("--out", default=str(common.DATA / "texse_split.json"))
    ap.add_argument("--probed", nargs="*", default=[str(common.PROJ / "out" / "texse_gemini.json")])
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jsonl", help="also write data/<prefix>_train.jsonl and data/<prefix>_heldout.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    forced = {x["id"] for p in args.probed if Path(p).exists() for x in json.load(open(p))["results"]}
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r["error_category"]].append(r)
    rng = random.Random(args.seed)
    train, ev = [], []
    for cat, rs in sorted(by_cat.items()):
        pre = [r for r in rs if r["id"] in forced]
        rest = [r for r in rs if r["id"] not in forced]
        want = min(max(round(args.frac * len(rs)), 1 if len(rs) >= 2 else 0), len(rs) - 1)
        rng.shuffle(rest)
        k = max(0, want - len(pre))
        ev += pre + rest[:k]
        train += rest[k:]
    if args.jsonl:
        for name, rs in (("train", train), ("heldout", ev)):
            with open(common.DATA / f"{args.jsonl}_{name}.jsonl", "w") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rs)
    both = sum(1 for rs in by_cat.values() if len(rs) >= 2)
    singles = sum(1 for rs in by_cat.values() if len(rs) == 1)
    json.dump({"train": [r["question_id"] for r in train], "eval": [r["question_id"] for r in ev]},
              open(args.out, "w"))
    print(f"{len(rows)} rows -> train {len(train)}, eval {len(ev)} ({len(ev) / len(rows):.0%}); "
          f"{len(forced & {r['id'] for r in ev})} probed rows in eval; {both} categories in both halves, "
          f"{singles} singleton categories in train only; wrote {args.out}")


if __name__ == "__main__":
    main()
