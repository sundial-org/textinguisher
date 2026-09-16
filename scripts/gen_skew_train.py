"""Organic multi-file skew TRAINING set: arXiv 2010-2020 projects (>= 2 tex/sty/bib files, \\input/\\include) that fail
under TeX Live 2025 (Modal farm) and compile under the TeX Live of their year (modal/modal_compile_hist.py). The historic
render (pdftotext + pages) is the pseudo-reference, so a fix that restores the intended document scores `strict`.

  gen_skew_train.py --target 3000 [--resume]  # -> data/skew_train.jsonl, data/refs_skew.jsonl, corpus/arxiv_skew_multi/
  (rows come straight from the dataset's parquet shards: the HF rows API 429s above ~15 requests/min)
  gen_skew_train.py --split                   # freeze data/skew_multi_heldout.jsonl (200 ids) + data/refs_skew.json
"""
import argparse
import hashlib
import io
import json
import random
import re
import shutil
import tarfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import common
import gen_skew
from gen_arxiv import materialize
from gen_data import V2_FULL_LINES, V2_MAX_CHARS
from gen_skew import PKG_EXT, fix_placeholders, year_of

OUT_ROOT = common.PROJ / "corpus" / "arxiv_skew_multi"
TRAIN, HELDOUT = common.DATA / "skew_train.jsonl", common.DATA / "skew_multi_heldout.jsonl"
REFS_JSONL, REFS = common.DATA / "refs_skew.jsonl", common.DATA / "refs_skew.json"
STATS = common.DATA / "skew_train_stats.json"
gen_skew.LOG = common.PROJ / "out" / "gen_skew_train.log"
log = gen_skew.log
HF = "https://huggingface.co/datasets/scholarweave/arxiv-latex/resolve/main/arxiv_part_{:04d}.parquet"
SHARDS = range(2, 12)  # manifest: part 0002 = 0412..1005, ..., 0011 = 2010..2101; row groups of 2048 rows, ~90 MB
YEARS, HIST_YEARS = range(2010, 2021), range(2013, 2021)
INPUT_RE = re.compile(r"\\(?:input|include)\{")
_hist_fns: dict[int, object] = {}


def open_shard(i: int):
    import fsspec
    import pyarrow.parquet as pq
    return pq.ParquetFile(fsspec.open(HF.format(i), "rb", block_size=4 << 20).open())


def fetch_group(shard: int, rg: int) -> list[dict]:
    """One parquet row group over HTTP range requests (~5 s), instead of the rate-limited rows API (25-100 rows/req)."""
    return open_shard(shard).read_row_group(rg, columns=["id", "license", "categories", "latex"]).to_pylist()


def is_multi(d: Path) -> bool:
    files = common.project_files(d)
    n = sum(f.endswith((".tex", ".sty", ".bib")) for f in files)
    return n >= 2 and bool(INPUT_RE.search(files.get("main.tex", "")))


def hist_compile(src: Path, year: int, engine: str) -> dict:
    """Render under texlive/texlive:TL<year>-historic (nearest deployed year). Returns {ok, hist_year, text, pages, err}."""
    import modal
    hy = min(max(year, HIST_YEARS[0]), HIST_YEARS[-1])
    if hy not in _hist_fns:
        _hist_fns[hy] = modal.Function.from_name("latexfix-compile-hist", f"compile_tl{hy}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(src, arcname=".")
    try:
        r = _hist_fns[hy].remote(buf.getvalue(), "main.tex", common.ENGINE_FLAG[engine], False, 150)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "hist_year": hy, "err": f"modal: {str(e)[:80]}"}
    errs = [it for it in common.parse_latex_log(r["log"], root_file="main.tex") if it["severity"] == "error"]
    ok = r["pdf_exists"] and not errs and not r["timed_out"]
    return {"ok": ok, "hist_year": hy, "text": " ".join((r["pdf_text"] or "").split()) if ok else None,
            "pages": r["pdf_pages"], "err": errs[0]["message"][:120] if errs else ("timeout" if r["timed_out"] else None)}


def process(d: Path, year: int, calib: bool) -> tuple[str, dict | None, dict | None]:
    """Compile under TL2025; failing projects go to the historic farm. Returns (verdict, row, ref)."""
    r = common.compile_project(d, [], timeout=120, backend="modal")
    if r["ok"]:
        if calib:  # how often does the SAME document score strict across TL versions? (ceiling for the reward)
            h = hist_compile(d, year, r["engine"])
            if h["ok"]:
                sim, sim_a = (common.pdf_similarity(r["pdf_text"], h["text"], asc) for asc in (False, True))
                return "calib", {"sim": sim, "sim_ascii": sim_a, "pages_ok": h["pages"] == r["pdf_pages"],
                                 "hist_year": h["hist_year"]}, None
        return "ok", None, None
    if r["timed_out"]:
        return "timeout", None, None
    if not r["error_items"]:
        return "fail-noerr", None, None
    files = common.project_files(d)
    err_lines = common.error_lines_from_items(r["items"], "main.tex", set(files) | {"main.tex"})
    first = r["error_items"][0]
    mf = gen_skew.missing_file(first)
    if not err_lines:
        return "fail-unlocated", None, None
    if mf and not mf.lower().endswith(PKG_EXT):
        return f"excluded-missing {mf}", None, None
    h = hist_compile(d, year, r["engine"])
    if not h["ok"]:
        return f"hist-fail TL{h['hist_year']} [{h['err']}]", None, None
    reported = err_lines[0]["file"] or "main.tex"
    slug = d.parent.name
    meta = json.loads((d.parent / "meta.json").read_text())
    row = {"id": f"arxiv_skew_multi_{slug}", "doc": f"arxiv_skew_multi/{slug}", "src_dir": str(d), "bundles": [],
           "mutation": "organic", "seed": 0, "lane": "project", "file_edits": [], "bib": False, "reported_file": reported,
           "fix_files": [], "error_lines": err_lines, "log_tail": r["log"][-common.MAX_LOG_CHARS:],
           "prompt": common.build_fix_prompt(reported, err_lines, r["log"], files.get(reported, ""),
                                             full_lines=V2_FULL_LINES, max_chars=V2_MAX_CHARS, tree=common.tree_block(d)),
           "engine": r["engine"], "signature": gen_skew.signature(first), "first_error": first["message"][:200],
           "missing_file": mf, "n_errors": len(r["error_items"]), "n_files": len(files), "arxiv_id": meta["id"],
           "year": year, "hist_year": h["hist_year"], "ref_pages": h["pages"], "license": meta.get("license")}
    return "KEEP", row, {"text": h["text"], "pages": h["pages"]}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def generate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    funnel, calib, done_groups = Counter(), [], []
    if args.resume:
        if STATS.exists():
            s = json.load(STATS.open()); funnel.update(s["funnel"]); calib = s.get("calib", []); done_groups = s.get("groups", [])
    else:
        for p in (TRAIN, HELDOUT, REFS_JSONL):
            p.unlink(missing_ok=True)
        shutil.rmtree(OUT_ROOT, ignore_errors=True); OUT_ROOT.mkdir()
    seen = {r["arxiv_id"] for r in load_rows(TRAIN) + load_rows(HELDOUT)}
    kept = len(seen)
    for p in OUT_ROOT.iterdir():  # projects left by an interrupted run
        if p.name not in {re.sub(r"[^A-Za-z0-9_.-]", "_", a) for a in seen}:
            shutil.rmtree(p, ignore_errors=True)
    seen |= {r["arxiv_id"] for r in load_rows(gen_skew.DATA)}  # keep skew_eval ids out of training
    groups = [[s, g] for s in SHARDS for g in range(open_shard(s).metadata.num_row_groups)]
    rng.shuffle(groups)
    groups = [g for g in groups if g not in done_groups]
    log(f"start: kept {kept}, {len(seen)} ids excluded, target {args.target}, {len(groups)} row groups left")
    t0, t_stat = time.monotonic(), time.monotonic()
    with ThreadPoolExecutor(args.fetchers) as fx, ThreadPoolExecutor(args.workers) as cx:
        fetches, compiles = {}, {}
        while fetches or compiles or kept < args.target:
            more = kept < args.target and funnel["sampled"] < args.max_sampled and (time.monotonic() - t0) < args.max_minutes * 60
            while more and groups and len(fetches) < args.fetchers and len(compiles) < 4 * args.workers:  # do not run ahead of the farm
                fetches[fx.submit(fetch_group, *groups[-1])] = groups.pop()
            if not fetches and not compiles:
                break
            done, _ = wait(set(fetches) | set(compiles), return_when=FIRST_COMPLETED)
            for f in done:
                if f in fetches:
                    grp = fetches.pop(f)
                    try:
                        rows = f.result()
                    except Exception as e:  # noqa: BLE001
                        log(f"fetch {grp} failed: {str(e)[:80]}"); groups.insert(0, grp); time.sleep(3); continue
                    done_groups.append(grp)
                    for row in rows:
                        funnel["sampled"] += 1
                        year = year_of(row["id"])
                        if year not in YEARS or row["id"] in seen:
                            continue
                        seen.add(row["id"])
                        funnel["in_years"] += 1
                        d = materialize(row, OUT_ROOT, args.max_chars)
                        if d and not is_multi(d):
                            shutil.rmtree(d.parent, ignore_errors=True); d = None
                        if d:
                            funnel["materialized"] += 1
                            fix_placeholders(d)
                            compiles[cx.submit(process, d, year, funnel["materialized"] % args.calib_every == 0)] = d
                    continue
                d = compiles.pop(f)
                try:
                    verdict, row, ref = f.result()
                except Exception as e:  # noqa: BLE001
                    verdict, row, ref = f"error {str(e)[-160:]}", None, None
                key = verdict.split()[0]
                funnel[{"ok": "compiled_ok", "calib": "compiled_ok", "timeout": "timed_out", "fail-noerr": "failed_no_error",
                        "fail-unlocated": "failed_unlocated", "excluded-missing": "excluded_missing_input",
                        "hist-fail": "hist_failed", "KEEP": "kept"}.get(key, key)] += 1
                if key in ("fail-unlocated", "excluded-missing", "hist-fail", "KEEP"):
                    funnel["compiled_failed"] += 1
                if key == "calib":
                    calib.append(row)
                if row and ref:
                    kept += 1
                    with TRAIN.open("a") as fo, REFS_JSONL.open("a") as fr:
                        fo.write(json.dumps(row) + "\n"); fr.write(json.dumps({row["doc"]: ref}) + "\n")
                    verdict = f"KEEP TL{row['hist_year']} [{row['signature'][:50]}] ({kept}/{args.target})"
                else:
                    shutil.rmtree(d.parent, ignore_errors=True)
                log(f"{d.parent.name:14s} {verdict}")
            if time.monotonic() - t_stat > 60:
                t_stat = time.monotonic(); el = (t_stat - t0) / 60
                log(f"progress: kept {kept}/{args.target} sampled {funnel['sampled']} mat {funnel['materialized']} "
                    f"fail {funnel['compiled_failed']} hist-fail {funnel['hist_failed']} {el:.0f}min, "
                    f"ETA ~{(args.target - kept) * el / max(kept, 1):.0f}min, in flight {len(compiles)}")
                STATS.write_text(json.dumps({"funnel": funnel, "calib": calib, "groups": done_groups}, indent=1))
    STATS.write_text(json.dumps({"funnel": funnel, "calib": calib, "groups": done_groups}, indent=1))
    log(f"done: {kept} rows -> {TRAIN}")


def split(args: argparse.Namespace) -> None:
    """Freeze 200 held-out ids (stable once written) and build refs_skew.json from the incremental jsonl."""
    rows = load_rows(TRAIN) + load_rows(HELDOUT)
    held = {r["arxiv_id"] for r in load_rows(HELDOUT)}
    for r in sorted(rows, key=lambda r: hashlib.sha1(r["arxiv_id"].encode()).hexdigest()):
        if len(held) >= args.heldout:
            break
        held.add(r["arxiv_id"])
    HELDOUT.write_text("".join(json.dumps(r) + "\n" for r in rows if r["arxiv_id"] in held))
    TRAIN.write_text("".join(json.dumps(r) + "\n" for r in rows if r["arxiv_id"] not in held))
    refs = {}
    for l in REFS_JSONL.read_text().splitlines():  # era renders differ in glyph->text mapping: ASCII compare + own floor
        refs.update({d: {**r, "ascii": True, "sim_min": args.sim_min} for d, r in json.loads(l).items()})
    REFS.write_text(json.dumps(refs))
    print(f"train {len(rows) - len(held)}, heldout {len(held)}, refs {len(refs)} -> {REFS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fetchers", type=int, default=2, help="parallel parquet row-group reads (2048 rows each)")
    ap.add_argument("--workers", type=int, default=96, help="parallel Modal compiles")
    ap.add_argument("--max-chars", type=int, default=300_000)
    ap.add_argument("--max-sampled", type=int, default=500_000)
    ap.add_argument("--max-minutes", type=float, default=290)
    ap.add_argument("--calib-every", type=int, default=6, help="every Nth compiled-ok project also renders historic")
    ap.add_argument("--heldout", type=int, default=200)
    ap.add_argument("--sim-min", type=float, default=0.93, help="strict floor stored in refs_skew.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--split", action="store_true")
    args = ap.parse_args()
    (split if args.split else generate)(args)


if __name__ == "__main__":
    main()
