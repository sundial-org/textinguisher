"""Re-sample the rows of an eval result whose response came back empty (reasoning ate the token cap) with a higher
cap, re-score them, and update the file in place (refilled rows get `refilled: true`; the file then mixes two budgets).
  EVAL_MAX_TOKENS=8000 refill_empty.py out/<set>_<tag>.json"""
import asyncio
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import common
from run_eval import sample_gateway, sample_tinker, score_response


def main() -> None:
    common.load_env()
    path = sys.argv[1]
    d = json.load(open(path))
    data = d["data"] if d["data"].startswith("/") else str(common.PROJ / "scripts" / d["data"]) if d["data"].startswith("..") else str(common.PROJ / d["data"])
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(data)}
    refs = {}
    for name in ("refs.json", "refs_pool.json", "refs_skew.json"):  # same merge as run_eval
        if (common.DATA / name).exists():
            refs.update(json.loads((common.DATA / name).read_text()))
    todo = [r for r in d["results"] if not r["response"].strip()]
    print(f"{path}: {len(todo)} empty of {len(d['results'])}", flush=True)
    if not todo:
        return
    arm = d["arm"]
    if arm.startswith("gateway:"):
        samples = asyncio.run(sample_gateway([rows[r["id"]] for r in todo], arm.split(":", 1)[1], 8))
    else:
        samples = sample_tinker([rows[r["id"]] for r in todo], arm.split(":", 1)[1] if arm.startswith("tinker:") and not arm.startswith("tinker://") else arm)
    with ProcessPoolExecutor(16) as ex:
        futs = [ex.submit(score_response, rows[r["id"]], s["response"], refs.get(rows[r["id"]]["doc"])) for r, s in zip(todo, samples)]
        for r, s, f in zip(todo, samples, futs):
            sc = f.result()
            for k in ("retry_doc", "retry_log_tail", "retry_error_lines"):
                sc.pop(k, None)
            r.update(s); r.update(sc); r["pass1"] = sc["compiled"]; r["refilled"] = True
    d["fixed"] = sum(r["compiled"] for r in d["results"]); d["strict"] = sum(r["strict"] for r in d["results"])
    json.dump(d, open(path, "w"), indent=1)
    print(f"refilled: still empty {sum(1 for r in d['results'] if not r['response'].strip())}; fixed {d['fixed']} strict {d['strict']}", flush=True)


if __name__ == "__main__":
    main()
