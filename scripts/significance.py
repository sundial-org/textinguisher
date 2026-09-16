"""Paired McNemar (exact, two-sided) between two eval runs on the same set.

usage: significance.py <a.json> <b.json> [metric]   # reports b's wins over a
"""
import json, sys
from math import comb


def load(path):
    return {r["id"]: r for r in json.load(open(path))["results"]}


def mcnemar(a, b, metric="compiled"):
    ids = sorted(set(a) & set(b))
    b_only = sum(1 for i in ids if not a[i].get(metric) and b[i].get(metric))
    a_only = sum(1 for i in ids if a[i].get(metric) and not b[i].get(metric))
    n, k = b_only + a_only, min(b_only, a_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return len(ids), b_only, a_only, p


if __name__ == "__main__":
    a, b, metric = sys.argv[1], sys.argv[2], (sys.argv[3:] or ["compiled"])[0]
    n, b_only, a_only, p = mcnemar(load(a), load(b), metric)
    print(f"n={n} {metric}: b-only={b_only} a-only={a_only} p={p:.4f}")
