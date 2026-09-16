"""Validation set for the v2 runs: 96 texse2_train + 24 skew_train rows that have a verified reference (REFS env), held
out of every v2 train mix by id (train_rl.py val=rl_val2.jsonl). rl_val.jsonl was 100% SFT-seen and had no multi-file rows."""
import json
import random

import common

refs = common.load_refs()
rng = random.Random(2)
out = []
for name, n in (("texse2_train.jsonl", 96), ("skew_train.jsonl", 24)):
    rows = [json.loads(l) for l in (common.DATA / name).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["doc"] in refs and len(r["prompt"]) <= 20000]
    out += rng.sample(rows, n)
(common.DATA / "rl_val2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(f"rl_val2.jsonl: {len(out)} rows")
