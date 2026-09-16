"""References v2: recompile every labeled doc on the current farm (TeX Live 2026) and keep text + pages + page rasters
(pdf_img), so the reward can score figures and layout, and so no training row keeps a reference that no longer compiles.
Organic arXiv skew rows are re-rendered on the historic farm (their reference is the era render, ascii text mode).
Unlabeled pool rows get no entry here: teach_refs.py / verify_refs.py build their references from cross-checked teacher fixes.

  build_refs2.py --data texse2_train.jsonl [--out refs2.json]        # labeled rows -> refs2.json
  build_refs2.py --data skew_train.jsonl --out refs2_skew.json        # organic skew rows -> historic farm
"""
import argparse
import io
import json
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import common

_hist = {}


def hist_render(src: Path, year: int, engine: str) -> dict | None:
    import modal
    hy = min(max(year, 2013), 2020)
    if hy not in _hist:
        _hist[hy] = modal.Function.from_name("latexfix-compile-hist", f"compile_tl{hy}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(src, arcname=".")
    call = _hist[hy].spawn(buf.getvalue(), "main.tex", common.ENGINE_FLAG[engine], False, 150)
    try:
        r = call.get(timeout=300)  # a lost worker must not hang the build
    except Exception:  # noqa: BLE001
        call.cancel()
        return None
    errs = [it for it in common.parse_latex_log(r["log"], root_file="main.tex") if it["severity"] == "error"]
    if not (r["pdf_exists"] and not errs and not r["timed_out"]):
        return None
    return {"text": " ".join((r["pdf_text"] or "").split()), "pages": r["pdf_pages"], "img": r.get("pdf_img"),
            "ascii": True, "sim_min": 0.93, "tl": f"TL{hy}"}


def ref_one(row: dict) -> tuple[str, dict | None]:
    try:
        return _ref_one(row)
    except Exception as e:  # noqa: BLE001  (a missing corpus dir must not kill the build)
        print(f"  {row['doc']}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        return row["doc"], None


def _ref_one(row: dict) -> tuple[str, dict | None]:
    if row.get("mutation") == "organic" and row.get("hist_year"):
        return row["doc"], hist_render(Path(row["src_dir"]), row["hist_year"], row.get("engine", "pdflatex"))
    r = common.compile_project(Path(row["src_dir"]), [], timeout=120, bib=bool(row.get("bib")), backend="modal")
    if not r["ok"]:
        return row["doc"], None
    return row["doc"], {"text": r["pdf_text"], "pages": r["pdf_pages"], "img": r.get("pdf_img"), "tl": r.get("texlive")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--out", default="refs2.json")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    common.load_env()
    out_path = common.DATA / args.out
    refs = json.loads(out_path.read_text()) if out_path.exists() else {}
    failed_path = out_path.with_suffix(".failed.json")
    failed = set() if args.retry_failed or not failed_path.exists() else set(json.loads(failed_path.read_text()))
    docs = {}
    for name in args.data:
        for l in (common.DATA / name).read_text().splitlines():
            r = json.loads(l)
            labeled = bool(r.get("break_edits") or r.get("break_edit") or r.get("file_edits")) or (r.get("mutation") == "organic" and r.get("hist_year"))
            if labeled and r["doc"] not in refs and r["doc"] not in failed:
                docs.setdefault(r["doc"], r)
    print(f"{len(refs)} refs cached, {len(failed)} known failures, {len(docs)} docs to compile -> {out_path}", flush=True)
    t0, n = time.monotonic(), 0
    with ThreadPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(ref_one, r) for r in docs.values()]):
            doc, ref = f.result()
            n += 1
            if ref:
                refs[doc] = ref
            else:
                failed.add(doc)
            if n % 200 == 0 or n == len(docs):
                out_path.write_text(json.dumps(refs))
                failed_path.write_text(json.dumps(sorted(failed)))
                print(f"[{n}/{len(docs)}] refs={len(refs)} failed={len(failed)} {time.monotonic() - t0:.0f}s", flush=True)
    print(f"done: {len(refs)} refs, {len(failed)} failed (listed in {failed_path.name})")


if __name__ == "__main__":
    main()
