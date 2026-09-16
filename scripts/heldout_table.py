"""Print compiles / compiles with <=40 chars removed / intact (also no structural element lost) / strict for every out/<set>_*.json arm, restricted to the ids
currently in data/<set>.jsonl (so re-drawn splits and partial runs compare on the same rows).
  heldout_table.py texse2_heldout"""
import glob
import json
import sys

import common
from fix_env import deleted_chars, structure_lost
from run_eval import broken_project, broken_text

name = sys.argv[1] if len(sys.argv) > 1 else "texse_heldout"
rows = {json.loads(l)["id"]: json.loads(l) for l in (common.DATA / f"{name}.jsonl").read_text().splitlines() if l.strip()}
ids = set(rows)
drop = common.DATA / f"{name}_drop.json"  # audited defective rows (broken reference, empty output, hidden fix)
if drop.exists():
    ids -= set(json.loads(drop.read_text()))


def intact(r):  # 40 chars is the reporting threshold; the RL reward's free allowance is fix_env.DELETE_FREE
    row, resp = rows[r["id"]], common.THINK_RE.sub("", r["response"], count=1)
    if row.get("lane") == "project":
        b = broken_project(row); f, err = common.apply_file_edits(b, common.parse_file_edits(resp))
        lost = err or sum(structure_lost(b[k], f[k]) for k in b if k in f)
    else:
        b = broken_text(row); f, err = common.apply_edits(b, common.parse_edits(resp))
        lost = err or structure_lost(b, f)
    return r["compiled"] and not lost and deleted_chars(r["response"]) <= 40

refs = common.load_refs()  # REFS env picks the reference set; sim/strict are reported on the rows that have one
ids &= {i for i in ids if rows[i]["doc"] in refs} if common.os.environ.get("REFS") else ids
table = []
for f in sorted(glob.glob(str(common.PROJ / "out" / f"{name}_*.json"))):
    if "_p2_" in f.split(f"{name}_")[-1]:  # results of the rebuilt-prompt variant of this set belong to <name>_p2
        continue
    res = [r for r in json.load(open(f))["results"] if r["id"] in ids]
    if not res:
        continue
    n = len(res)
    sims = [r.get("sim") or 0.0 for r in res if r.get("sim") is not None or not r["compiled"]]
    table.append((f.split(f"{name}_")[-1][:-5], n, 100 * sum(r["compiled"] for r in res) / n,
                 100 * sum(1 for r in res if r["compiled"] and deleted_chars(r["response"]) <= 40) / n,
                 100 * sum(intact(r) for r in res) / n, 100 * sum(r["strict"] for r in res) / n,
                 100 * sum(1 for r in res if (r.get("sim") or 0) >= 0.95) / n, 100 * sum(sims) / max(1, len(sims)),
                 sum(r["out_tokens"] or 0 for r in res) / n))
table.sort(key=lambda r: -r[2])
print(f"{name}: {len(ids)} ids\n{'arm':36s} {'n':>4s} {'compiles':>8s} {'del<=40':>8s} {'intact':>7s} {'strict':>7s} {'sim>=.95':>8s} {'meansim':>7s} {'tok':>5s}")
for a, n, c, d, i, s, s95, ms, t in table:
    print(f"{a[:36]:36s} {n:4d} {c:8.1f} {d:8.1f} {i:7.1f} {s:7.1f} {s95:8.1f} {ms:7.1f} {t:5.0f}")
