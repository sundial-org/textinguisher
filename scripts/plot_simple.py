"""Simple decision figure: content kept (and exact in the label) vs cost per 1,000 fixes and vs latency, five frontier models and our three
options. Frontier cost = tokens x list price (plot_cost.PRICE); ours = 8xH200 at $0.001261/s/GPU over measured throughput ($1.10/1k single
call), retry = +one call on the failing 16%, best-of-16 ~ 16 sampled outputs. Latency: frontier p50 measured; ours single p50 measured;
retry/best-of-16 estimated (mean). Writes out/simple.png."""
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

import common
from fix_env import deleted_chars

matplotlib.use("Agg")
PRICE = {"gpt-6-astra": (10, 50), "claude-fable-5.1": (10, 50), "gpt-5.5": (5, 30), "gemini-3.8-flash": (0.75, 3.75), "claude-sonnet-5": (2, 10)}
NAMES = {"gpt-6-astra": "GPT-6 Astra", "claude-fable-5.1": "Claude Fable 5.1", "gpt-5.5": "GPT-5.5", "gemini-3.8-flash": "Gemini 3.8 Flash", "claude-sonnet-5": "Claude Sonnet 5"}
refs = json.loads((common.DATA / "refs2_heldout.json").read_text()); drop = set(json.loads((common.DATA / "texse2_heldout_drop.json").read_text()))
rows = {json.loads(l)["id"]: json.loads(l) for l in (common.DATA / "texse2_heldout.jsonl").read_text().splitlines()}
ids = [i for i in rows if rows[i]["doc"] in refs and i not in drop]
in_tok = {i: (len(common.SYSTEM) + len(rows[i]["prompt"])) / 4 for i in ids}


def load(tag):
    return {r["id"]: r for r in json.load(open(common.PROJ / "out" / f"texse2_heldout_{tag}.json"))["results"]}


def kept_exact(d):
    n = len(ids)
    return (100 * sum(d[i]["compiled"] and deleted_chars(common.THINK_RE.sub("", d[i]["response"], count=1)) <= 40 for i in ids) / n,
            100 * sum(d[i]["strict"] for i in ids) / n)


def lat(tag):
    w = sorted(r["wall_s"] for r in json.load(open(common.PROJ / "out" / f"lat15_{tag}.json"))["results"] if r.get("wall_s"))
    return w[len(w) // 2]


pts = []
for tag, (pi, po) in PRICE.items():
    d = load(tag)
    cost = sum(in_tok[i] * pi + (d[i]["out_tokens"] or 0) * po for i in ids) / len(ids) / 1e6 * 1000
    pts.append((NAMES[tag], cost, lat(tag), *kept_exact(d), "frontier"))
single = 1.10
pts += [("ours, one call (0.75 s measured)", single, 0.75, *kept_exact(load("rlM6c_000120")), "ours"),
        ("ours + one compiler retry (est.)", single * 1.16, 1.05, *kept_exact(load("rlR1_finalrc_bo1")), "ours"),
        ("ours, best-of-16 + retry (est.)", single * 12, 2.5, *kept_exact(load("rlF2_finalr_bo16")), "loop")]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, xi, xl in ((axes[0], 1, "$ per 1,000 fixes (log)"), (axes[1], 2, "seconds per fix (log)")):
    for name, cost, l, kept, exact, kind in pts:
        x = cost if xi == 1 else l
        style = dict(color="#1f5fbf", s=90) if kind == "ours" else dict(color="#1f5fbf", s=90, facecolors="none") if kind == "loop" else dict(color="#8a949e", s=50)
        ax.scatter(x, kept, zorder=3, **style)
        ax.annotate(f"{name}\nexact {exact:.0f}%", (x, kept), textcoords="offset points", xytext=(6, -4), fontsize=8, color="#1f2933" if kind != "frontier" else "#6b7480")
    ax.set_xscale("log"); ax.set_xlabel(xl); ax.grid(alpha=.3)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_ylabel("content kept, % (compiles without deleting content)"); axes[0].set_xticks([1, 3, 10, 30]); axes[1].set_xticks([0.5, 1, 2, 4])
axes[0].set_ylim(45, 97)
fig.suptitle("Verified TeX.SE errors (255). Filled blue = recommended; hollow = the 16-sample loop; grey = frontier APIs")
fig.tight_layout(); fig.savefig(common.PROJ / "out" / "simple.png", dpi=150)
for name, cost, l, kept, exact, kind in pts:
    print(f"{name:36s} ${cost:5.2f}/1k  {l:4.2f}s  kept {kept:4.1f}  exact {exact:4.1f}")
