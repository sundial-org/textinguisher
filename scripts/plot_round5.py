"""Round-5 chart: exact fix and content kept vs latency on the verified 255, frontier models (measured p50) and our variants
(single greedy measured 0.75 s; compiler-loop variants estimated: + model call and one ~1.1 s compile on the failing share, best-of-k adds
one batched sample call and two parallel compile waves). Writes out/round5.png.  plot_round5.py <single_tag> <retry_tag> <bo_tag> [bo_label]"""
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

import common
from fix_env import deleted_chars
matplotlib.use("Agg")
ARMS = [("gpt-6-astra", "GPT-6 Astra"), ("claude-fable-5.1", "Claude Fable 5.1"), ("gpt-5.5", "GPT-5.5"), ("grok-4.6", "Grok 4.6"),
        ("claude-opus-5", "Claude Opus 5"), ("gemini-3.8-flash", "Gemini 3.8 Flash"), ("gemini-3.7-flash", "Gemini 3.7 Flash"),
        ("claude-sonnet-5", "Claude Sonnet 5"), ("claude-haiku-4.5", "Claude Haiku 4.5"), ("gpt-5-mini", "GPT-5 mini")]
refs = json.loads((common.DATA / "refs2_heldout.json").read_text())
drop = set(json.loads((common.DATA / "texse2_heldout_drop.json").read_text()))
rows = {json.loads(l)["id"]: json.loads(l) for l in (common.DATA / "texse2_heldout.jsonl").read_text().splitlines()}
ids = [i for i in rows if rows[i]["doc"] in refs and i not in drop]


def metrics(tag):  # same definitions as plot_headline_clean / heldout_table
    p = common.PROJ / "out" / f"texse2_heldout_{tag}.json"
    if not p.exists():
        return None
    d = {r["id"]: r for r in json.loads(p.read_text())["results"]}
    n = len(ids)
    return (100 * sum(d[i]["compiled"] for i in ids) / n,
            100 * sum(d[i]["compiled"] and deleted_chars(common.THINK_RE.sub("", d[i]["response"], count=1)) <= 40 for i in ids) / n,
            100 * sum(d[i]["strict"] for i in ids) / n)


def latency(tag):
    p = common.PROJ / "out" / f"lat15_{tag}.json"
    if not p.exists():
        return None
    w = sorted(r["wall_s"] for r in json.loads(p.read_text())["results"] if r.get("wall_s"))
    return w[len(w) // 2]


if __name__ == "__main__":
    single, retry, bo = sys.argv[1:4]
    bo_label = sys.argv[4] if len(sys.argv) > 4 else "ours, best-of-16 + retry"
    pts = []
    for tag, label in ARMS:
        m, lat = metrics(tag), latency(tag)
        if m and lat:
            pts.append((label, lat, m, "frontier"))
    m1, m2, m3 = metrics(single), metrics(retry), metrics(bo)
    fail = 1 - m1[0] / 100
    pts += [("ours, one call", 0.75, m1, "ours"), ("ours + one compiler retry", round(0.75 + fail * (0.75 + 1.1), 2), m2, "ours"),
            (bo_label, round(1.0 + 1.1 + fail * 1.9, 1), m3, "ours")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, idx, name in ((axes[0], 2, "exact fix (PDF matches the human fix), %"), (axes[1], 1, "content kept (compiles, <= 40 chars deleted), %")):
        for label, lat, m, kind in pts:
            c = "#1f5fbf" if kind == "ours" else "#8a949e"
            ax.scatter(lat, m[idx], s=70 if kind == "ours" else 45, color=c, zorder=3)
            ax.annotate(label, (lat, m[idx]), textcoords="offset points", xytext=(5, 4), fontsize=8, color="#1f2933" if kind == "ours" else "#5c6670")
        ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 4, 8]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlabel("seconds per fix (log; ours with the loop = estimate)"); ax.set_ylabel(name); ax.grid(alpha=.3)
    fig.suptitle("Verified TeX.SE errors (255): our model alone and with the compiler in the loop vs frontier APIs")
    fig.tight_layout(); fig.savefig(common.PROJ / "out" / "round5.png", dpi=150)
    print("\n".join(f"{l:28s} {lat:5.2f}s  compile {m[0]:.1f} kept {m[1]:.1f} exact {m[2]:.1f}" for l, lat, m, _ in pts))
