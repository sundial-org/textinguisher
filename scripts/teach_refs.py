"""Multi-teacher fix candidates for rows without a human reference (pool, organic skew, unlabeled github).
Every teacher answers the production prompt at T=0; every candidate is applied and compiled on the farm and stored with
its PDF text, page count and page rasters, so verify_refs.py can accept a reference only when independent teachers agree
on the output (or a calibrated judge clears the lone survivor). Output: data/teach_<stem>.jsonl, one line per row, resumable.

  teach_refs.py --data texse_pool.jsonl --teachers google/gemini-3.8-flash openai/gpt-5.5
  teach_refs.py --data texse_pool.jsonl --teachers openai/gpt-6-astra --only-ids ids.json   # third opinion where needed
"""
import argparse
import asyncio
import difflib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common
from fix_env import deleted_chars
from run_eval import broken_project, broken_text, sample_gateway


def compile_candidate(row: dict, response: str) -> dict:
    """Apply the edits and compile; keep everything verify_refs.py needs to compare candidates."""
    response = common.THINK_RE.sub("", response, count=1)
    out = {"applied": False, "compiled": False, "deleted": deleted_chars(response), "diff_lines": None,
           "text": None, "pages": None, "img": None, "error": None}
    if row.get("lane") == "project":
        files = broken_project(row)
        patched, err = common.apply_file_edits(files, common.parse_file_edits(response))
        if err:
            out["error"] = err[:120]
            return out
        overrides = {f: patched[f] for f in patched if patched[f] != files.get(f)}
        if not overrides:
            out["error"] = "no-op"
            return out
        out["diff_lines"] = sum(len([l for l in difflib.unified_diff(files[f].splitlines(), patched[f].splitlines(), lineterm="")
                                     if l[:1] in "+-" and l[:3] not in ("+++", "---")]) for f in overrides)
        r = common.compile_project(Path(row["src_dir"]), [], overrides=overrides, timeout=120, bib=row.get("bib", False), backend="modal")
    else:
        doc = broken_text(row)
        edits = common.parse_edits(response) or [(s, r) for _, s, r in common.parse_file_edits(response)]
        patched, err = common.apply_edits(doc, edits)
        if err:
            out["error"] = err[:120]
            return out
        out["diff_lines"] = sum(1 for l in difflib.unified_diff(doc.splitlines(), patched.splitlines(), lineterm="")
                                if l[:1] in "+-" and l[:3] not in ("+++", "---"))
        r = common.compile_project(Path(row["src_dir"]), [], overrides={"main.tex": patched}, timeout=120, backend="modal")
    out["applied"] = True
    out["compiled"] = r["ok"]
    if r["ok"]:
        out.update(text=r["pdf_text"], pages=r["pdf_pages"], img=r.get("pdf_img"), tl=r.get("texlive"))
    else:
        out["error"] = (r["error_items"][0]["message"][:120] if r["error_items"] else "timeout" if r["timed_out"] else "no pdf")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--only-ids", default=None, help="json list of row ids (e.g. rows still lacking an agreed reference)")
    ap.add_argument("--skip-docs", default=None, help="refs json: rows whose doc already has a reference are skipped")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    common.load_env()
    rows = [json.loads(l) for l in (common.DATA / args.data).read_text().splitlines() if l.strip()]
    if args.only_ids:
        keep = set(json.loads(Path(args.only_ids).read_text()))
        rows = [r for r in rows if r["id"] in keep]
    if args.skip_docs:
        have = set(json.loads((common.DATA / args.skip_docs).read_text()))
        rows = [r for r in rows if r["doc"] not in have]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out) if args.out else common.DATA / f"teach_{Path(args.data).stem}.jsonl"
    done: dict[str, dict] = {}
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                rec = json.loads(l)
                done[rec["id"]] = rec
    todo = [r for r in rows if not all(t in done.get(r["id"], {}).get("cands", {}) for t in args.teachers)]
    print(f"{len(rows)} rows, {len(todo)} need {args.teachers} -> {out}", flush=True)
    t0 = time.monotonic()
    loop = asyncio.new_event_loop()  # one loop for every gateway batch (a fresh asyncio.run per batch closes the previous client's loop)
    with out.open("a") as fh, ThreadPoolExecutor(64) as ex:
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            for teacher in args.teachers:
                need = [r for r in chunk if teacher not in done.get(r["id"], {}).get("cands", {})]
                if not need:
                    continue
                got = loop.run_until_complete(sample_gateway(need, teacher, args.concurrency))
                futs = [ex.submit(compile_candidate, r, s["response"]) for r, s in zip(need, got)]
                for r, s, f in zip(need, got, futs):
                    rec = done.setdefault(r["id"], {"id": r["id"], "doc": r["doc"], "cands": {}})
                    try:
                        res = f.result(timeout=420)
                    except Exception as e:  # noqa: BLE001  (a hung farm call must not stall the shard)
                        res = {"applied": False, "compiled": False, "deleted": 0, "diff_lines": None, "text": None, "pages": None, "img": None, "error": f"hung: {type(e).__name__}"}
                    rec["cands"][teacher] = {"response": s["response"], "sample_error": s.get("sample_error"), **res}
            for r in chunk:  # append the row's current record (last record per id wins on reload)
                fh.write(json.dumps(done[r["id"]]) + "\n")
            fh.flush()
            n = c + len(chunk)
            comp = sum(any(v["compiled"] for v in done[r["id"]]["cands"].values()) for r in todo[:n])
            el = time.monotonic() - t0
            print(f"[{n}/{len(todo)}] rows with a compiling candidate={comp} ({comp / n:.0%}) {el / 60:.1f}min ETA {el / n * (len(todo) - n) / 60:.0f}min", flush=True)


if __name__ == "__main__":
    main()
