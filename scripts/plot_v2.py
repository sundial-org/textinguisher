"""Bar chart of the v2 round on the clean single-file benchmark (rows with a verified human reference, audited defects
dropped): compiles / strict / near-exact (sim >= 0.95) per arm.  plot_v2.py [tag=label ...] -> out/v2_clean.png"""
import json
import sys

import matplotlib
import matplotlib.pyplot as plt

import common

matplotlib.use("Agg")
ARMS = [a.split("=", 1) for a in sys.argv[1:]] or [
    ("gpt-6-astra", "Astra"), ("claude-fable-5.1", "Fable"), ("gpt-5.5", "GPT-5.5"), ("rlM6_000150", "M6 (old data)"),
    ("rlC0_000150", "C0 clean data"), ("rlS1_000120", "S1 sim^2"), ("rlS2_000150", "S2 sim^4")]
refs = json.loads((common.DATA / "refs2_heldout.json").read_text())
drop = set(json.loads((common.DATA / "texse2_heldout_drop.json").read_text()))
rows = {json.loads(l)["id"]: json.loads(l) for l in (common.DATA / "texse2_heldout.jsonl").read_text().splitlines()}
ids = [i for i in rows if rows[i]["doc"] in refs and i not in drop]
vals = []
for tag, label in ARMS:
    p = common.PROJ / "out" / f"texse2_heldout_{tag}.json"
    if not p.exists():
        continue
    d = {r["id"]: r for r in json.loads(p.read_text())["results"]}
    n = len(ids)
    vals.append((label, 100 * sum(d[i]["compiled"] for i in ids) / n, 100 * sum(d[i]["strict"] for i in ids) / n,
                 100 * sum((d[i].get("sim") or 0) >= 0.95 for i in ids) / n))
fig, ax = plt.subplots(figsize=(9, 4.2))
w = 0.27
for k, (name, col) in enumerate([("compiles", "#bbbbbb"), ("strict", "#1f77b4"), ("near-exact (sim >= .95)", "#ff7f0e")]):
    ax.bar([i + (k - 1) * w for i in range(len(vals))], [v[k + 1] for v in vals], w, label=name, color=col)
ax.set_xticks(range(len(vals)), [v[0] for v in vals], rotation=15)
ax.set_ylabel("% of clean rows"); ax.set_ylim(0, 100); ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"Clean single-file benchmark ({len(ids)} rows with a verified human reference)")
fig.tight_layout(); fig.savefig(common.PROJ / "out" / "v2_clean.png", dpi=150)
print("\n".join(f"{v[0]:16s} {v[1]:5.1f} {v[2]:5.1f} {v[3]:5.1f}" for v in vals))
