"""Re-score saved eval results (out/<set>_<tag>.json) against the current references (REFS env) without re-sampling:
recompiles every stored response on the farm and rewrites compiled/strict/pdf_sim/text_sim/img_sim/sim in place.
Rows whose doc has no reference keep compiled but get strict=False, sim=None (the tables report them as unscorable).

  REFS=refs2.json,refs2_heldout.json rescore.py out/texse2_heldout_gpt-6-astra.json [...]
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common
from run_eval import score_response


def main(paths: list[str]) -> None:
    common.load_env()
    refs = common.load_refs()
    for p in paths:
        p = Path(p)
        d = json.loads(p.read_text())
        rows = {json.loads(l)["id"]: json.loads(l) for l in (common.DATA.parent / d["data"]).read_text().splitlines() if l.strip()}
        res = d["results"]
        with ThreadPoolExecutor(int(common.os.environ.get("RESCORE_THREADS", "48"))) as ex:
            scored = list(ex.map(lambda r: score_response(rows[r["id"]], r["response"], refs.get(rows[r["id"]]["doc"])), res))
        for r, s in zip(res, scored):
            r.update({k: s.get(k) for k in ("applied", "compiled", "exact", "diff_lines", "pdf_sim", "text_sim", "img_sim", "sim", "pages_ok", "strict")})
            r["has_ref"] = rows[r["id"]]["doc"] in refs
        d["fixed"], d["strict"] = sum(r["compiled"] for r in res), sum(r["strict"] for r in res)
        d["refs"] = common.os.environ.get("REFS", common.REFS_DEFAULT)
        p.write_text(json.dumps(d, indent=1))
        n_ref = sum(r["has_ref"] for r in res)
        sims = [r["sim"] for r in res if r["compiled"] and r["sim"] is not None]
        print(f"{p.name}: compiled {d['fixed']}/{len(res)}, strict {d['strict']} (refs on {n_ref}), mean sim of compiled {sum(sims) / max(1, len(sims)):.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
