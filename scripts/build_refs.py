"""Compile each doc referenced by a data file once; cache ground-truth PDF text + pages (failed docs are not cached, so
they are retried next run).

Usage: build_refs.py --data ../data/hard_eval.jsonl [--out ../data/refs.json]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import common


def ref_one(doc: str, src_dir: str, bundles: list[str], bib: bool = False) -> tuple[str, dict | None]:
    r = common.compile_project(Path(src_dir), [Path(b) for b in bundles], timeout=90, bib=bib)
    if not r["ok"]:
        return doc, None
    return doc, {"text": r["pdf_text"], "pages": r["pdf_pages"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(common.DATA / "refs.json"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    docs = {r["doc"]: (r["src_dir"], r["bundles"], bool(r.get("bib"))) for r in rows}  # bib rows are scored with bibtex
    out_path = Path(args.out)
    refs = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = {d: v for d, v in docs.items() if d not in refs}
    print(f"{len(docs)} docs, {len(todo)} to compile")
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(ref_one, d, s, b, bib) for d, (s, b, bib) in todo.items()]
        for f in as_completed(futs):
            doc, ref = f.result()
            if ref:
                refs[doc] = ref
            print(f"  {doc}: {'ok' if ref else 'FAILED'}")
    out_path.write_text(json.dumps(refs))
    print(f"wrote {out_path} ({len(refs)} refs)")


if __name__ == "__main__":
    main()
