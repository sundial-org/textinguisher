"""Rank evaluated RL checkpoints for the multi-file model of record: score = compiles on multi2_heldout + skew_multi_heldout
(270 rows, context prompts), gate = texse_heldout compiles >= GATE287 (default 70.4 = the 7b model minus one point). usage: select_ckpt.py rlM2 rlM3 ..."""
import json
import sys

import common

GATE287 = float(common.os.environ.get("GATE287", "70.4"))
STEPS = ["000030", "000060", "000090", "000120", "000150", "final"]


def cnt(f):
    try:
        r = json.load(open(f))["results"]
        return sum(bool(x["compiled"]) for x in r), len(r)
    except Exception:  # noqa: BLE001
        return 0, 0


rows = []
for tag in sys.argv[1:]:
    for s in STEPS:
        m, sk, t = (cnt(common.PROJ / "out" / f"{n}_{tag}_{s}.json") for n in ("multi2_heldout", "skew_multi_heldout", "texse_heldout"))
        if not (m[1] and sk[1] and t[1]):
            continue
        multi, single = 100 * (m[0] + sk[0]) / (m[1] + sk[1]), 100 * t[0] / t[1]
        rows.append((multi, single, tag, s, m, sk))
for multi, single, tag, s, m, sk in sorted(rows, reverse=True):
    ok = "ok " if single >= GATE287 else "gate"
    print(f"{ok} {tag}_{s} multi={multi:.1f} (real {m[0]}/{m[1]}, skew {sk[0]}/{sk[1]}) single287={single:.1f}")
