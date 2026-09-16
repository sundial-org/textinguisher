"""Decision scatter for the model choice: every candidate system on the verified 255 (compiles / kept / exact) against latency, plus
exact vs kept. Frontier latencies are measured p50; ours: single call measured (0.75 s), loop variants estimated (+0.75 s model call and
a 1.1 s median compile on the failing share; best-of-k = one batched sample call + two parallel compile waves). Writes out/decision.png."""
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

import common
from plot_round5 import ARMS, latency, metrics  # noqa: E402  (module-level code in plot_round5 only defines helpers before argv use)

matplotlib.use("Agg")
OURS = [  # tag, label, latency (s), measured?
    ("rlM6c_000120", "M6c_120 single (shipped candidate)", 0.75, True),
    ("rlM6c_000120_r1", "M6c_120 + 1 retry", 1.05, False),
    ("rlR1_final", "R1 two-turn, single", 0.75, False),
    ("rlR1_finalrc_bo1", "R1 + 1 retry", 1.1, False),
    ("rlR1_finalrc_bo8", "R1 + best-of-8 + retry", 2.4, False),
    ("rlF2_000120", "F2_120 single", 0.75, False),
    ("rlF2_final", "F2 final, single", 0.75, False),
    ("rlF2_finalr_bo1", "F2 final + 1 retry", 1.2, False),
    ("rlF2_finalr_bo16", "F2 final + best-of-16 + retry", 2.5, False),
    ("rlF2_finalr_bo32", "F2 final + best-of-32 + retry", 2.7, False),
]
pts = [(label, latency(tag), metrics(tag), "frontier") for tag, label in ARMS if metrics(tag) and latency(tag)]
pts += [(label, lat, metrics(tag), "ours") for tag, label, lat, _ in OURS if metrics(tag)]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
panels = [(axes[0], lambda l, m: (l, m[2]), "seconds per fix (log)", "exact fix, %"),
          (axes[1], lambda l, m: (l, m[1]), "seconds per fix (log)", "content kept, %"),
          (axes[2], lambda l, m: (m[1], m[2]), "content kept, %", "exact fix, %")]
for ax, f, xl, yl in panels:
    for label, lat, m, kind in pts:
        x, y = f(lat, m)
        ours = kind == "ours"
        ax.scatter(x, y, s=70 if ours else 40, color="#1f5fbf" if ours else "#8a949e", zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7, color="#1f2933" if ours else "#6b7480")
    if xl.startswith("seconds"):
        ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 4, 8]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=.3)
fig.suptitle("Verified TeX.SE errors (255). Blue = ours; only the single-call latency is measured, loop latencies are estimates")
fig.tight_layout(); fig.savefig(common.PROJ / "out" / "decision.png", dpi=150)
for label, lat, m, kind in pts:
    if kind == "ours":
        print(f"{label:36s} {lat:4.2f}s  compile {m[0]:5.1f} kept {m[1]:5.1f} exact {m[2]:5.1f}")
