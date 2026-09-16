"""Results figure: real TeX.SE held-out (headline) | v1 speed-vs-accuracy Pareto | hard-fair bars | agentic bars.

Reads out/texse_heldout_*.json, out/acc_*.json + out/lat_*.json (v1), out/hard3_*.json, out/agent_full_*.json.
--lat-override ARM_SUBSTR=SECONDS replaces an arm's latency (e.g. real Modal serving). Writes out/pareto.png.
"""
import argparse
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from fix_env import deleted_chars  # noqa: E402

LABELS = {
    "claude-haiku-4.5": ("Haiku 4.5 (prod today)", "tab:orange"),
    "claude-sonnet-5": ("Sonnet 5", "tab:orange"),
    "claude-fable-5.1": ("Fable 5.1", "tab:orange"),
    "gpt-5.2": ("GPT-5.2", "tab:green"),
    "gpt-5-mini": ("GPT-5 mini", "tab:green"),
    "gpt-5-nano": ("GPT-5 nano", "tab:green"),
    "gemini-3.7-flash": ("Gemini 3.7 Flash", "tab:blue"),
    "gemini-3.5-flash-lite": ("Gemini 3.5 Flash Lite", "tab:blue"),
    "thinkingmachines/Inkling-Small": ("Inkling-Small (base)", "tab:red"),
    "59544ae4": ("Inkling-Small SFT r1", "tab:red"),
    "4ec160b2": ("Inkling-Small SFT r2", "tab:red"),
    "709bd70d": ("Inkling-Small SFT r3", "tab:red"),
    "6f14269d": ("Inkling-Small SFT r4", "tab:red"),
    "0f8e0cea": ("Inkling-Small SFT r5", "tab:red"),
    "4cbf404c": ("Inkling-Small SFT r6 (real data)", "tab:red"),
    "f313370b": ("SFT r5 + GRPO run 1", "tab:purple"),
    "85d6a73d": ("SFT r6 + GRPO run 2", "tab:purple"),
    "a0660e5e": ("SFT r6 + GRPO run 2", "tab:purple"),
    "6b3052b4": ("SFT r6 + GRPO run 7 (3x RL data) = model of record", "tab:purple"),
}
HIDE_HELDOUT = re.compile(r"_(r4e1|r5|r5e0|r7|r8|rl20|rl50|rl70|rl[2367]_0\d+)\.json$")  # intermediate checkpoints, kept for the curve only
OFFSETS = {"Gemini 3.5 Flash Lite": (-6, 10), "Inkling-Small SFT r2": (10, -22), "Inkling-Small SFT r6 (real data)": (10, -8), "SFT r6 + GRPO run 2": (10, 8), "Inkling-Small SFT r3": (10, 8), "GPT-5.2": (8, -13)}


def label_of(arm: str) -> tuple[str, str]:
    for key, v in LABELS.items():
        if key in arm:
            name, color = v
            if "epoch1" in arm:
                name += " (ep1)"
            if m := re.search(r"sampler_weights/0*(\d+)$", arm):
                name += f" @{m.group(1)}"
            if "compile-only" in arm or "f313370b" in arm:
                name += " (compile-only reward)"
            return name, color
    return arm.split("/")[-1][:24], "tab:gray"


def median_wall(results):
    walls = sorted(r["wall_s"] for r in results if r.get("wall_s") is not None)
    return walls[len(walls) // 2] if walls else None


def bars(ax, names, series, title, ylabel="% fixed"):
    """Grouped bars; series = [(label, values, color)]."""
    w = 0.8 / len(series)
    for j, (lab, vals, color) in enumerate(series):
        xs = [i + (j - (len(series) - 1) / 2) * w for i in range(len(names))]
        ax.bar(xs, vals, w, label=lab, color=color)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    for lab in ax.get_xticklabels():
        if "SFT" in lab.get_text() or "GRPO" in lab.get_text():
            lab.set_fontweight("bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat-override", action="append", default=[], help="ARM_SUBSTR=SECONDS")
    args = ap.parse_args()
    overrides = {k: float(v) for k, v in (o.split("=") for o in args.lat_override)}
    out_dir = common.PROJ / "out"

    texse = [json.loads(p.read_text()) for p in sorted(out_dir.glob("texse_heldout_*.json")) if not HIDE_HELDOUT.search(p.name)]
    acc = {d["arm"]: d for d in (json.loads(p.read_text()) for p in out_dir.glob("acc_*.json"))}
    lat = {d["arm"]: median_wall(d["results"]) for d in (json.loads(p.read_text()) for p in out_dir.glob("lat_*.json"))}
    hard = [json.loads(p.read_text()) for p in sorted(out_dir.glob("hard3_*.json"))]
    agent = [json.loads(p.read_text()) for p in sorted(out_dir.glob("agent_full_*.json"))]

    fig, (tx, ax, bx, cx) = plt.subplots(1, 4, figsize=(27, 6.5), gridspec_kw={"width_ratios": [1.15, 1.1, 1, 0.85]})

    # 1. real errors: compile rate, compile with <=40 chars of content removed (anti "simplify until it compiles"), strict
    texse.sort(key=lambda d: -d["fixed"])
    n = texse[0]["n"]
    bars(tx, [label_of(d["arm"])[0] for d in texse],
         [("compiles", [100 * d["fixed"] / n for d in texse], "tab:gray"),
          ("compiles, <=40 chars removed", [100 * sum(1 for r in d["results"] if r["compiled"] and deleted_chars(r["response"]) <= 40) / n for d in texse], "tab:red"),
          ("strict (PDF matches accepted answer)", [100 * d["strict"] / n for d in texse], "black")],
         f"REAL user errors: TeX.StackExchange held-out, n={n}")

    # 2. v1 speed vs accuracy
    for arm, d in acc.items():
        label, color = label_of(arm)
        x = next((s for k, s in overrides.items() if k in arm), None) or lat.get(arm) or median_wall(d["results"])
        y = 100 * d["fixed"] / d["n"]
        star = "SFT" in label
        ax.scatter(x, y, s=260 if star else 70, c=color, marker="*" if star else "o", zorder=3)
        ax.annotate(label + (" (real serving)" if any(k in arm for k in overrides) else ""), (x, y),
                    textcoords="offset points", xytext=OFFSETS.get(label, (8, 4)), fontsize=8.5, fontweight="bold" if star else None)
    ax.set_xscale("log")
    ax.set_xlabel("median seconds per fix (log)")
    ax.set_ylabel("% fixed (recompiles clean)")
    ax.set_title("v1 synthetic: single-error, 150 held-out docs")
    ax.grid(alpha=0.3)

    # 3. hard-fair synthetic
    hard.sort(key=lambda d: -d["fixed"])
    bars(bx, [label_of(d["arm"])[0] for d in hard],
         [("compiles", [100 * d["fixed"] / d["n"] for d in hard], "tab:gray"),
          ("strict (PDF matches)", [100 * d["strict"] / d["n"] for d in hard], "tab:red")],
         f"hard-fair synthetic: 2-4 stacked errors, n={hard[0]['n'] if hard else 0}")

    # 4. agentic lane
    agent.sort(key=lambda d: -d["strict"])
    dn = [max(1, d.get("n_done") or len(d["results"])) for d in agent]
    bars(cx, [label_of(d["arm"])[0] + f"\n({d.get('protocol', '')})" for d in agent],
         [("compiles", [100 * d["compiled"] / k for d, k in zip(agent, dn)], "tab:gray"),
          ("strict", [100 * d["strict"] / k for d, k in zip(agent, dn)], "tab:red")],
         f"agentic lane: tools + recompile, n={agent[0].get('n_done') or agent[0]['n'] if agent else 0}")
    for i, (d, k) in enumerate(zip(agent, dn)):
        cx.text(i, 4, f"{sum(r.get('out_tokens') or 0 for r in d['results']) / k:.0f} tok", ha="center", fontsize=7.5, color="white")

    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=170)
    print(f"wrote {out_dir / 'pareto.png'}")


if __name__ == "__main__":
    main()
