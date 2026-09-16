"""Pseudo-references for unlabeled pool rows: compile the compile-verified teacher fix (distill.py output; when several
files fix the same row the LAST file listed wins, so put the preferred teacher last) and store its PDF text/pages, so
the strict bonus on pool rows in RL measures agreement with that teacher, not correctness.
Writes data/refs_pool.json {doc: {text, pages}} (merging into it when it exists).
  build_pseudo_refs.py [pool.jsonl distill1.jsonl ...]"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import common
from fix_env import deleted_chars
from run_eval import broken_text

POOL, *DISTILL = sys.argv[1:] or ["texse_pool.jsonl", "distill_texse_pool_gemini-3-7-flash.jsonl",
                                  "distill_texse_pool_gemfail_claude-fable-5-1.jsonl"]
rows = {json.loads(l)["id"]: json.loads(l) for l in (common.DATA / POOL).read_text().splitlines()}
fixes: dict[str, str] = {}
for name in DISTILL:
    for l in (common.DATA / name).read_text().splitlines():
        r = json.loads(l)
        if r["ok"] and deleted_chars(r["target"]) <= 40 and (r.get("diff_lines") or 0) <= 30:
            fixes[r["id"]] = r["target"]


def one(rid: str) -> tuple[str, dict | None]:
    row = rows[rid]
    patched, err = common.apply_edits(broken_text(row), common.parse_edits(fixes[rid]))
    if err:
        return row["doc"], None
    res = common.compile_project(Path(row["src_dir"]), [], overrides={"main.tex": patched}, timeout=90)
    return row["doc"], {"text": res["pdf_text"], "pages": res["pdf_pages"]} if res["ok"] else None


if __name__ == "__main__":
    out = common.DATA / "refs_pool.json"
    refs, n = (json.loads(out.read_text()) if out.exists() else {}), 0
    with ProcessPoolExecutor(24) as ex:
        for f in as_completed([ex.submit(one, rid) for rid in fixes]):
            doc, ref = f.result()
            n += 1
            if ref:
                refs[doc] = ref
            if n % 100 == 0:
                print(f"{n}/{len(fixes)} refs={len(refs)}", flush=True)
    out.write_text(json.dumps(refs))
    print(f"refs_pool.json now {len(refs)} pseudo-refs ({len(fixes)} teacher fixes processed)")
