"""Honest-benchmark figures on the rows whose human reference was confirmed by an independent teacher or the judge
(data/refs2_heldout.json, audited defects dropped). Per model: compiles, compiles without deleting content (<= 40 content chars
removed), exact fix (strict). Writes out/headline_clean.png (latency vs exact-fix rate, hollow marker = compile rate, the drop =
compiles that changed the PDF) and out/hackshare.png (each model's compiles split into faithful vs content-deleting).
  plot_headline_clean.py [ours_tag]   (default rlM6_000150 = the deployed checkpoint)"""
import json
import sys

import matplotlib
import matplotlib.pyplot as plt

import common
from fix_env import deleted_chars

matplotlib.use("Agg")
ARMS = [("gpt-6-astra", "GPT-6 Astra"), ("claude-fable-5.1", "Claude Fable 5.1"), ("gpt-5.5", "GPT-5.5"), ("grok-4.6", "Grok 4.6"),
        ("claude-opus-5", "Claude Opus 5"), ("gemini-3.8-flash", "Gemini 3.8 Flash"), ("gemini-3.7-flash", "Gemini 3.7 Flash"),
        ("claude-sonnet-5", "Claude Sonnet 5"), ("claude-haiku-4.5", "Claude Haiku 4.5"), ("gpt-5-mini", "GPT-5 mini")]
OURS_TAG = sys.argv[1] if len(sys.argv) > 1 else "rlM6_000150"
ACCENT, INK, MUTED, RED = "#1f5fbf", "#1f2933", "#8a949e", "#c0392b"

refs = json.loads((common.DATA / "refs2_heldout.json").read_text())
drop = set(json.loads((common.DATA / "texse2_heldout_drop.json").read_text()))
rows = {json.loads(l)["id"]: json.loads(l) for l in (common.DATA / "texse2_heldout.jsonl").read_text().splitlines()}
ids = [i for i in rows if rows[i]["doc"] in refs and i not in drop]
ours = json.loads((common.PROJ / "out" / "ours.json").read_text())


def metrics(tag):
    p = common.PROJ / "out" / f"texse2_heldout_{tag}.json"
    if not p.exists():
        return None
    d = {r["id"]: r for r in json.loads(p.read_text())["results"]}
    n = len(ids)
    comp = 100 * sum(d[i]["compiled"] for i in ids) / n
    kept = 100 * sum(d[i]["compiled"] and deleted_chars(common.THINK_RE.sub("", d[i]["response"], count=1)) <= 40 for i in ids) / n
    strict = 100 * sum(d[i]["strict"] for i in ids) / n
    return comp, kept, strict


def latency(tag):
    p = common.PROJ / "out" / f"lat15_{tag}.json"
    if not p.exists():
        return None
    w = sorted(r["wall_s"] for r in json.loads(p.read_text())["results"] if r.get("wall_s"))
    return w[len(w) // 2]


rowsout = []
for tag, label in ARMS:
    m, lat = metrics(tag), latency(tag)
    if m and lat:
        rowsout.append((label, lat, *m, False))
m = metrics(OURS_TAG)
rowsout.append(("Inkling-Small, fine-tuned (ours)", ours["latency_p50"], *m, True))

# 1. latency vs exact fix, with the compile ceiling as a hollow marker and the drop as a line
fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=200)
fig.patch.set_facecolor("white")
OFFSET = {"GPT-6 Astra": (9, 4), "GPT-5.5": (9, -6), "Claude Opus 5": (9, -6), "Claude Fable 5.1": (9, 4), "Grok 4.6": (9, 4),
          "Gemini 3.8 Flash": (9, -6), "Gemini 3.7 Flash": (9, 4), "Claude Sonnet 5": (9, 4), "Claude Haiku 4.5": (9, 4), "GPT-5 mini": (9, 4)}
for label, lat, comp, kept, strict, is_ours in rowsout:
    col = ACCENT if is_ours else MUTED
    ax.plot([lat, lat], [strict, comp], color=col, linewidth=1.2, alpha=0.6, zorder=2)
    ax.scatter(lat, comp, s=70, facecolors="white", edgecolors=col, linewidths=1.3, zorder=3)
    ax.scatter(lat, strict, s=150 if is_ours else 100, color=col, zorder=4, linewidths=0)
    dx, dy = (10, -8) if is_ours else OFFSET.get(label, (9, -4))
    ax.annotate(label, (lat, strict), xytext=(dx, dy), textcoords="offset points", fontsize=10.5 if is_ours else 10, color=col,
                fontweight="bold" if is_ours else "normal", va="top")
ax.scatter([], [], s=70, facecolors="white", edgecolors=MUTED, linewidths=1.3, label="compiles")
ax.scatter([], [], s=100, color=MUTED, label="compiles and the PDF is the intended one (exact fix)")
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.set_xlim(0, max(6, max(r[1] for r in rowsout) + 1)); ax.set_ylim(20, 100)
ax.set_xlabel("seconds per fix", fontsize=10.5, color=INK); ax.set_ylabel("% of verified real errors", fontsize=10.5, color=INK)
ax.set_title("Compile rate is the ceiling; the filled marker is what the user actually gets", loc="left", fontsize=12.5, color=INK, pad=14)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#d0d5da")
ax.tick_params(colors=MUTED, labelsize=9); ax.grid(axis="y", color="#e6e9ec", linewidth=0.8); ax.set_axisbelow(True)
fig.text(0.01, 0.005, f"{len(ids)} held-out real errors whose human fix was confirmed by an independent model or a judge; exact = same pages and >= 98.5% word agreement with that fix. Median latency, same client.", fontsize=7.5, color=MUTED)
fig.tight_layout(); fig.savefig(common.PROJ / "out" / "headline_clean.png")

# 2. compiles split into faithful vs content-deleting, one bar per model
fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=200)
fig.patch.set_facecolor("white")
order = sorted(rowsout, key=lambda r: -r[2])
labels = [r[0] for r in order]
kept, dele, strict = [r[3] for r in order], [r[2] - r[3] for r in order], [r[4] for r in order]
y = range(len(order))
ax.barh(y, kept, color=[ACCENT if r[5] else "#b9c3cc" for r in order], label="compiles, content kept")
ax.barh(y, dele, left=kept, color=RED, alpha=0.85, label="compiles by deleting > 40 chars of content")
ax.scatter(strict, list(y), marker="|", s=260, color=INK, zorder=5, label="exact fix")
ax.set_yticks(list(y), labels, fontsize=9.5); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("% of verified real errors", fontsize=10.5, color=INK)
ax.set_title("Where each model's compile rate comes from", loc="left", fontsize=12.5, color=INK, pad=12)
ax.legend(loc="lower right", fontsize=9, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=9); ax.grid(axis="x", color="#e6e9ec", linewidth=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(common.PROJ / "out" / "hackshare.png")
print(f"{'model':34s} {'compiles':>8s} {'kept':>6s} {'exact':>6s} {'sec':>5s}")
for label, lat, comp, kept_, strict_, _ in sorted(rowsout, key=lambda r: -r[4]):
    print(f"{label:34s} {comp:8.1f} {kept_:6.1f} {strict_:6.1f} {lat:5.2f}")
print("rows", len(ids))
