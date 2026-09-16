"""Headline figure: real-error fix rate vs end-to-end latency, one point per model. Writes out/headline.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json

import common

# accuracy: % of the 669 held-out real errors whose fix compiles (out/texse2_heldout_<tag>.json);
# latency: median wall s over the same 15 prompts from this client (out/lat15_<tag>.json); ours from modal/REPORT.md bench.
ARMS = [("gpt-6-astra", "GPT-6 Astra"), ("claude-fable-5.1", "Claude Fable 5.1"), ("gpt-5.5", "GPT-5.5"), ("grok-4.6", "Grok 4.6"), ("claude-opus-5", "Claude Opus 5"),
        ("gemini-3.8-flash", "Gemini 3.8 Flash"), ("gemini-3.7-flash", "Gemini 3.7 Flash"), ("claude-sonnet-5", "Claude Sonnet 5"),
        ("claude-haiku-4.5", "Claude Haiku 4.5"), ("gpt-5-mini", "GPT-5 mini")]  # the models in the post's table
ids = {json.loads(l)["id"] for l in (common.DATA / "texse2_heldout.jsonl").read_text().splitlines()}
OFFSET = {"GPT-6 Astra": (9, 16), "GPT-5.5": (9, -10), "Claude Opus 5": (9, -12), "Claude Fable 5.1": (-9, 12), "Grok 4.6": (9, -10),
          "Gemini 3.8 Flash": (9, -10)}  # label nudges (points)
POINTS = []
for tag, label in ARMS:
    acc, lat = common.PROJ / "out" / f"texse2_heldout_{tag}.json", common.PROJ / "out" / f"lat15_{tag}.json"
    if acc.exists() and lat.exists():
        res = [r for r in json.load(open(acc))["results"] if r["id"] in ids]
        walls = sorted(r["wall_s"] for r in json.load(open(lat))["results"] if r.get("wall_s"))
        POINTS.append((label, walls[len(walls) // 2], 100 * sum(r["compiled"] for r in res) / len(res)))
_o = json.loads((common.PROJ / "out" / "ours.json").read_text())  # written by out/finalize.sh: the deployed checkpoint's bench + 669-set accuracy
OURS = ("Inkling-Small, fine-tuned (ours)", _o["latency_p50"], _o["texse2_compiles"])
ACCENT, INK, MUTED = "#1f5fbf", "#1f2933", "#8a949e"

fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=200)
fig.patch.set_facecolor("white")
for name, x, y in POINTS:
    ax.scatter(x, y, s=110, color=MUTED, zorder=3, linewidths=0)
    dx, dy = OFFSET.get(name, (9, -4))
    ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=10, color=INK, va="top", ha="left" if dx >= 0 else "right")
ax.scatter(OURS[1], OURS[2], s=170, color=ACCENT, zorder=4, linewidths=0)
ax.annotate(OURS[0], (OURS[1], OURS[2]), xytext=(10, -8), textcoords="offset points", fontsize=10.5, color=ACCENT, fontweight="bold", va="top")
ax.set_xlim(0, max(6, max(x for _, x, _ in POINTS) + 1)); ax.set_ylim(30, 91)
ax.set_xlabel("seconds per fix", fontsize=10.5, color=INK)
ax.set_ylabel("real errors fixed (%)", fontsize=10.5, color=INK)
ax.set_title("A 12B-active model fixes real LaTeX errors at frontier accuracy, in under a second", loc="left", fontsize=12.5, color=INK, pad=14)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#d0d5da")
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(axis="y", color="#e6e9ec", linewidth=0.8); ax.set_axisbelow(True)
fig.text(0.01, 0.005, "669 held-out real errors from TeX.StackExchange; fixed = patched document recompiles clean. Median end-to-end latency, same client, streaming; API models at default settings.", fontsize=7.5, color=MUTED)
fig.tight_layout()
fig.savefig("out/headline.png"); print("wrote out/headline.png")
