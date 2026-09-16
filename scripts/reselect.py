"""Re-run candidate selection offline on a passk --dump file and print compiles / kept / exact for several rules (no re-sampling).
  reselect.py out/cands_rlM6c_000120v_bo8.jsonl"""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
n = len(rows)
RULES = {
    "consensus": lambda c: (-c["votes"], c["deleted"]),
    "verifier": lambda c: (-(c.get("p_yes") or 0), -c["votes"], c["deleted"]),
    "verifier+kept": lambda c: (c["deleted"] > 40, -(c.get("p_yes") or 0), -c["votes"], c["deleted"]),
    "consensus+kept": lambda c: (c["deleted"] > 40, -c["votes"], c["deleted"]),
    "votes+2p": lambda c: (-(c["votes"] + 2 * (c.get("p_yes") or 0)), c["deleted"]),
}
print(f"{sys.argv[1]}: {n} prompts; oracle exact {100 * sum(any(c['compiled'] and c['strict'] for c in r['cands']) for r in rows) / n:.1f}, "
      f"oracle kept {100 * sum(any(c['compiled'] and c['deleted'] <= 40 for c in r['cands']) for r in rows) / n:.1f}")
for name, key in RULES.items():
    comp = kept = exact = 0
    for r in rows:
        cands = [c for c in r["cands"] if c["compiled"]]
        if not cands:
            continue
        s = min(cands, key=key)
        comp += 1; kept += s["deleted"] <= 40; exact += s["strict"]
    print(f"{name:16s} {100 * comp / n:5.1f} {100 * kept / n:5.1f} {100 * exact / n:5.1f}")
