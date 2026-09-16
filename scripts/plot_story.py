"""Two supporting figures for the post: (1) real-error accuracy at each stage of the project,
(2) RL training curves on the held-out set for run 1 (compile-only reward, learned to delete), run 2 and run 7.
Reads out/texse_heldout_*.json. Writes out/story_stages.png, out/story_rl.png."""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
from fix_env import deleted_chars

ACCENT, INK, MUTED, LIGHT = "#1f5fbf", "#1f2933", "#8a949e", "#c9d3dd"
ids = {json.loads(l)["id"] for l in (common.DATA / "texse_heldout.jsonl").read_text().splitlines()}


def score(tag):
    res = [r for r in json.load(open(common.PROJ / "out" / f"texse_heldout_{tag}.json"))["results"] if r["id"] in ids]
    n = len(res)
    return 100 * sum(r["compiled"] for r in res) / n, 100 * sum(1 for r in res if r["compiled"] and deleted_chars(r["response"]) <= 40) / n


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d0d5da")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e6e9ec", linewidth=0.8); ax.set_axisbelow(True)


# 1. stages
stages = [("base model", 25.0), ("synthetic SFT", 15.0), ("real SFT", "r4e2"), ("+ teacher fixes", "r6"),
          ("RL", "rl2_final"), ("RL, 3x prompts", "rl7b_final"), ("+ multi-file", "rlM6_000150")]
vals = [t if isinstance(t, float) else score(t)[0] for _, t in stages]  # constants: base at the 2k cap, synthetic-only SFT probe (RESULTS.md)
fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=200)
colors = [LIGHT, "#e07b7b", LIGHT, LIGHT, LIGHT, LIGHT, ACCENT]  # the shipped stage in accent
ax.bar(range(len(vals)), vals, 0.62, color=colors, linewidth=0)
for i, v in enumerate(vals):
    ax.text(i, v + 1.2, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
ax.set_xticks(range(len(vals))); ax.set_xticklabels([s for s, _ in stages], fontsize=8.5, color=INK)
ax.set_ylim(0, 85); ax.set_ylabel("real errors fixed (%)", fontsize=10.5, color=INK)
ax.set_title("Synthetic errors made the model worse; real data and the compiler made it good", loc="left", fontsize=12, color=INK, pad=12)
style(ax)
fig.text(0.01, 0.005, "287 held-out real errors from TeX.StackExchange (original split); fixed = patched document recompiles clean", fontsize=7.5, color=MUTED)
fig.tight_layout(); fig.savefig("out/story_stages.png")

# 2. RL curves
runs = {"run 1: compile-only reward": [("rl20", 20), ("rl50", 50), ("rl70", 70), ("rl90", 90)],
        "run 2: + deletion cap, + PDF-match bonus": [(f"rl2_{s:06d}", s) for s in (30, 60, 90, 120, 150, 180, 210)] + [("rl2_final", 224)],
        "run 7: run 2 recipe, 3x prompts": [(f"rl7_{s:06d}", s) for s in (30, 60, 90, 120, 150, 180, 210)] + [("rl7_final", 224)]}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=200, sharey=True)
for (name, pts), color in zip(runs.items(), ("#e07b7b", MUTED, ACCENT)):
    have = [(s, score(t)) for t, s in pts if glob.glob(str(common.PROJ / "out" / f"texse_heldout_{t}.json"))]
    xs = [0] + [s for s, _ in have]
    base = score("r6") if "run 1" not in name else score("0f8e0cea" if False else "r5")
    for ax, k, lab in ((axes[0], 0, "fix compiles"), (axes[1], 1, "compiles and removes <=40 chars")):
        ys = [base[k]] + [sc[k] for _, sc in have]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4, label=name)
        ax.set_title(lab, loc="left", fontsize=11, color=INK); ax.set_xlabel("RL steps", fontsize=10, color=INK); style(ax)
axes[0].set_ylabel("real errors fixed (%)", fontsize=10.5, color=INK); axes[0].set_ylim(25, 80)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
fig.suptitle("What the compiler rewards: run 1 climbed by deleting content; the capped reward climbs cleanly", x=0.01, ha="left", fontsize=12, color=INK)
fig.text(0.01, 0.005, "287 held-out real errors; step 0 = the SFT checkpoint each run started from", fontsize=7.5, color=MUTED)
fig.tight_layout(); fig.savefig("out/story_rl.png"); print("wrote out/story_stages.png out/story_rl.png")
