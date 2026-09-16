"""Cost figure: dollars per 1,000 fix requests vs real-error accuracy (669 held-out). API cost = tokens x list price
(input tokens estimated as chars/4 of system+prompt; output tokens from the API usage, thinking included where billed).
Ours = 8xH200 on Modal ($0.001261/s per GPU) / measured throughput (modal/bench_throughput_c64.json). Writes out/cost.png."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

PRICE = {  # $ per 1M input, output tokens; list prices fetched 2026-09-07; same models as the post's table
    "gpt-6-astra": (10, 50), "claude-fable-5.1": (10, 50), "gpt-5.5": (5, 30), "grok-4.6": (2, 6), "claude-opus-5": (5, 25), "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75), "claude-sonnet-5": (2, 10), "claude-haiku-4.5": (1, 5), "gpt-5-mini": (0.25, 2)}
NAMES = {"claude-fable-5.1": "Claude Fable 5.1", "claude-sonnet-5": "Claude Sonnet 5", "claude-haiku-4.5": "Claude Haiku 4.5",
         "gemini-3.7-flash": "Gemini 3.7 Flash", "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite", "gpt-5.2": "GPT-5.2",
         "gpt-5-mini": "GPT-5 mini", "gpt-5-nano": "GPT-5 nano", "claude-opus-5": "Claude Opus 5", "gpt-5.5": "GPT-5.5", "gpt-6-astra": "GPT-6 Astra",
         "gpt-5.4-mini": "GPT-5.4 mini", "gpt-5.4-nano": "GPT-5.4 nano", "gemini-3.8-flash": "Gemini 3.8 Flash",
         "deepseek-v4-pro": "DeepSeek V4 Pro", "deepseek-v4-flash": "DeepSeek V4 Flash", "kimi-k3": "Kimi K3",
         "qwen3.8-max": "Qwen 3.8 Max", "grok-4.6": "Grok 4.6", "glm-5.3": "GLM 5.3", "mistral-large-3": "Mistral Large 3"}
ACCENT, INK, MUTED = "#1f5fbf", "#1f2933", "#8a949e"
rows = {json.loads(l)["id"]: json.loads(l) for l in (common.DATA / "texse2_heldout.jsonl").read_text().splitlines()}
in_tok = {i: (len(common.SYSTEM) + len(r["prompt"])) / 4 for i, r in rows.items()}
OFFSET = {"GPT-6 Astra": (9, 12), "Claude Opus 5": (9, -12), "Grok 4.6": (9, 12), "GPT-5.5": (-9, 12), "Gemini 3.8 Flash": (9, -12), "Claude Fable 5.1": (9, -4),
          "Claude Haiku 4.5": (9, -10), "GPT-5 mini": (9, 12)}  # label nudges (points)
pts = []
for tag, (pi, po) in PRICE.items():
    f = common.PROJ / "out" / f"texse2_heldout_{tag}.json"
    if not f.exists():
        continue
    res = [r for r in json.load(open(f))["results"] if r["id"] in rows]
    cost = sum(in_tok[r["id"]] * pi + (r["out_tokens"] or 0) * po for r in res) / len(res) / 1e6 * 1000
    pts.append((NAMES[tag], cost, 100 * sum(r["compiled"] for r in res) / len(res)))
_o = json.loads((common.PROJ / "out" / "ours.json").read_text())  # written by out/finalize.sh
tp = json.load(open(common.PROJ / "modal" / os.environ.get("TP_FILE", _o["throughput_file"])))
ours = ("Inkling-Small, fine-tuned\n(dedicated 8xH200)", 8 * 0.001261 * 3600 / tp["fixes_per_hour"] * 1000, _o["texse2_compiles"])

fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=200)
for name, x, y in pts:
    ax.scatter(x, y, s=110, color=MUTED, linewidths=0, zorder=3)
    dx, dy = OFFSET.get(name, (9, -4))
    ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=10, color=INK, va="top", ha="left" if dx >= 0 else "right")
ax.scatter(ours[1], ours[2], s=170, color=ACCENT, linewidths=0, zorder=4)
ax.annotate(ours[0], (ours[1], ours[2]), xytext=(10, 8), textcoords="offset points", fontsize=10.5, color=ACCENT, fontweight="bold", va="bottom")
ax.set_xscale("log"); ax.set_xlim(0.4, max(x for _, x, _ in pts) * 1.6); ax.set_ylim(15, 91)  # costs span two orders of magnitude
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
ax.set_xlabel("dollars per 1,000 fixes", fontsize=10.5, color=INK); ax.set_ylabel("real errors fixed (%)", fontsize=10.5, color=INK)
ax.set_title("Cost per fix: about a dollar per thousand", loc="left", fontsize=12.5, color=INK, pad=14)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#d0d5da")
ax.tick_params(colors=MUTED, labelsize=9); ax.grid(axis="y", color="#e6e9ec", linewidth=0.8); ax.set_axisbelow(True)
fig.text(0.01, 0.005, f"669 held-out real errors. API: list prices x measured tokens per request. Ours: 8xH200 at list price, "
         f"{tp['fixes_per_hour']:,.0f} fixes/hour measured at {tp['concurrency']} concurrent requests.", fontsize=7.5, color=MUTED)
fig.tight_layout(); fig.savefig(common.PROJ / "out" / "cost.png"); print("wrote out/cost.png", [(n, round(c, 2)) for n, c, _ in pts], round(ours[1], 2))
