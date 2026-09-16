"""Build references only from fixes with evidence behind them. Candidates for a row = the human fix (labeled rows, compiled
by build_refs2.py) + every teacher fix (teach_refs.py). A reference is accepted when
  agree  - two independent candidates compile to the same PDF (same page count, word similarity >= AGREE_SIM, raster
           similarity >= AGREE_IMG) and the chosen one deleted <= DEL_MAX content chars (human > smaller edit first);
  judge  - otherwise the compiling candidates go to the calibrated judge (judge.py) in that order; first faithful wins;
  drop   - nothing compiles or nothing clears.
Accepted answers on TeX.SE often rewrite the document (about half carry unrelated changes), so a lone human fix is
evidence, not ground truth. --human-only (benchmarks): keep the human reference when it agrees with a teacher or the
judge clears it, never substitute a teacher fix, so the benchmark stays anchored on human answers.
Writes data/<out> {doc: {text, pages, img, teacher, how, response}} and data/<out>.dropped.json {id: reason}.

  verify_refs.py --teach teach_texse_pool.jsonl --rows texse_pool.jsonl --out refs2_pool.json
  verify_refs.py --teach teach_texse2_heldout.jsonl --rows texse2_heldout.jsonl --out refs2_heldout.json --human-only
"""
import argparse
import asyncio
import json
from itertools import combinations

import common
from fix_env import deleted_chars

AGREE_SIM, AGREE_IMG, DEL_MAX = 0.985, 0.90, 40


def agree(a: dict, b: dict) -> bool:
    if a["pages"] != b["pages"]:
        return False
    if (common.pdf_similarity(a["text"], b["text"]) or 0) < AGREE_SIM:
        return False
    img = common.image_similarity(a.get("img"), b.get("img"))
    return img is None or img >= AGREE_IMG


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teach", required=True, nargs="+")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--human-refs", default="refs2.json")
    ap.add_argument("--human-only", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()
    common.load_env()
    from judge import JUDGE_MODEL, judge_many

    rows = {json.loads(l)["id"]: json.loads(l) for l in (common.DATA / args.rows).read_text().splitlines() if l.strip()}
    human = json.loads((common.DATA / args.human_refs).read_text()) if (common.DATA / args.human_refs).exists() else {}
    recs: dict[str, dict] = {}
    for name in args.teach:
        for l in (common.DATA / name).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                recs.setdefault(r["id"], {"id": r["id"], "doc": r["doc"], "cands": {}})["cands"].update(r["cands"])
    for rid, row in rows.items():  # the human fix is one more candidate
        if row.get("target") and row["doc"] in human:
            h = human[row["doc"]]
            recs.setdefault(rid, {"id": rid, "doc": row["doc"], "cands": {}})["cands"]["human"] = {
                "response": row["target"], "applied": True, "compiled": True, "deleted": deleted_chars(row["target"]),
                "diff_lines": 0, "text": h["text"], "pages": h["pages"], "img": h.get("img"), "tl": h.get("tl")}
    out_path, drop_path = common.DATA / args.out, common.DATA / (args.out + ".dropped.json")
    refs = json.loads(out_path.read_text()) if out_path.exists() else {}
    dropped: dict[str, str] = json.loads(drop_path.read_text()) if drop_path.exists() else {}  # keep every set's verdicts
    to_judge: list[tuple[str, str, dict]] = []

    def order(tc):  # human first, then the smaller edit
        return (tc[0] != "human", tc[1]["deleted"], tc[1]["diff_lines"] or 0)

    def accept(doc, t, c, how, reason=None):
        refs[doc] = {"text": c["text"], "pages": c["pages"], "img": c["img"], "teacher": t, "how": how, "response": c["response"],
                     "tl": c.get("tl"), **({"reason": reason} if reason else {})}

    for rid, rec in recs.items():
        if rec["doc"] in refs or rid not in rows:
            continue
        ok = {t: c for t, c in rec["cands"].items() if c["compiled"] and c["text"] is not None}
        if not ok:
            dropped[rid] = "no compiling candidate: " + "; ".join(f"{t.split('/')[-1]}={c.get('error')}" for t, c in rec["cands"].items())
            continue
        agreed = {t for t1, t2 in combinations(ok, 2) if agree(ok[t1], ok[t2]) for t in (t1, t2)}
        pick = sorted(((t, ok[t]) for t in agreed if ok[t]["deleted"] <= DEL_MAX), key=order)
        if args.human_only:
            pick = [tc for tc in pick if tc[0] == "human"]
        if pick:
            accept(rec["doc"], pick[0][0], pick[0][1], "agree")
            continue
        cands = sorted(ok.items(), key=order)
        if args.human_only:
            cands = [tc for tc in cands if tc[0] == "human"]
        if not cands:
            dropped[rid] = "human fix disagrees with every teacher" if "human" in ok else "no human fix compiles"
            continue
        to_judge += [(rid, t, c) for t, c in cands]
    n_agree = sum(v.get("how") == "agree" for v in refs.values())
    print(f"{len(recs)} rows: agreed {n_agree}, dropped {len(dropped)}, {len({j[0] for j in to_judge})} rows -> judge ({len(to_judge)} candidates)", flush=True)
    if to_judge and not args.no_judge:
        verdicts = asyncio.run(judge_many([(rows[rid], c["response"], c["text"]) for rid, _, c in to_judge], args.judge_model or JUDGE_MODEL))
        by_row: dict[str, list] = {}
        for (rid, t, c), v in zip(to_judge, verdicts):
            by_row.setdefault(rid, []).append((t, c, v))
        for rid, tcv in by_row.items():
            for t, c, v in tcv:
                if v["verdict"] == "faithful" and c["deleted"] <= DEL_MAX * 2:
                    accept(recs[rid]["doc"], t, c, "judge", v["reason"])
                    break
            else:
                dropped[rid] = "judge: " + "; ".join(f"{t.split('/')[-1]}={v['verdict']} ({v['reason'][:80]})" for t, _, v in tcv)
    elif to_judge:
        for rid in {j[0] for j in to_judge}:
            dropped[rid] = "disagree (judge skipped)"
    out_path.write_text(json.dumps(refs))
    drop_path.write_text(json.dumps(dropped, indent=1))
    hows = {h: sum(v.get("how") == h for v in refs.values()) for h in ("agree", "judge")}
    who = {t: sum(v.get("teacher") == t for v in refs.values()) for t in {v.get("teacher") for v in refs.values()}}
    print(f"-> {out_path.name}: {len(refs)} refs {hows} by {who}; dropped {len(dropped)} -> {drop_path.name}")


if __name__ == "__main__":
    main()
