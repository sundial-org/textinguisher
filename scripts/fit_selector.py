"""Reference-free selector for best-of-k: pick the ranking rule (over candidate features) that maximises exact fixes on the FIT dump,
then report it on the benchmark dump. Rules rank the compiling candidates by a tuple of (sign, feature) keys; no reference is used at
selection time.  fit_selector.py out/cands_fit_<tag>_bo16.jsonl out/cands_<tag>_bo16.jsonl"""
import itertools
import json
import sys

FEATS = ["votes", "deleted", "diff_lines", "len", "n_edits", "greedy"]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def pick(cands, rule):
    comp = [c for c in cands if c["compiled"]]
    if not comp:
        return None
    return min(comp, key=lambda c: tuple(-c[f] if sgn > 0 else (c[f] or 0) for sgn, f in rule) + (c["i"],))


def score(rows, rule):
    sel = [pick(r["cands"], rule) for r in rows]
    n = len(rows)
    return sum(bool(s and s["strict"]) for s in sel) / n, sum(bool(s and s["compiled"]) for s in sel) / n


fit, test = load(sys.argv[1]), load(sys.argv[2])
rules = []
for depth in (1, 2, 3):
    for feats in itertools.permutations(FEATS, depth):
        for signs in itertools.product((1, -1), repeat=depth):
            rules.append(tuple(zip(signs, feats)))
base = ((1, "votes"), (-1, "deleted"))
print(f"fit {len(fit)} rows, test {len(test)} rows, {len(rules)} rules; baseline (votes desc, deleted asc): fit exact {score(fit, base)[0]:.3f} test exact {score(test, base)[0]:.3f}")
print(f"oracle exact: fit {sum(any(c['strict'] for c in r['cands']) for r in fit) / len(fit):.3f} test {sum(any(c['strict'] for c in r['cands']) for r in test) / len(test):.3f}")
ranked = sorted(rules, key=lambda r: -score(fit, r)[0])
for r in ranked[:8]:
    fe, fc = score(fit, r); te, tc = score(test, r)
    print(f"  fit exact {fe:.3f} comp {fc:.3f} | TEST exact {te:.3f} comp {tc:.3f}  rule {[(('+' if s > 0 else '-') + f) for s, f in r]}")
