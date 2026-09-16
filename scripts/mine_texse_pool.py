"""Mine REAL broken LaTeX documents (no fix labels) from the TeX.StackExchange dump for RL with a
compile reward: any question (accepted answer not required) with a pasted error + a full-document
MWE that fails locally with a located error in main.tex.

  ../.venv/bin/python -u scripts/mine_texse_pool.py [--limit N] [--workers 6] [--resume]
Outputs: data/texse_pool.jsonl, data/texse_pool_report.md, corpus/texse_pool/<qid>/files/main.tex.
"""
import argparse
import hashlib
import json
import random
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import common
import mine_texse as mt
from gen_data import V2_FULL_LINES, V2_MAX_CHARS

POOL = common.PROJ / "corpus" / "texse_pool"
CANDIDATES = POOL / "candidates.jsonl"
PARSE_STATS = POOL / "parse_stats.json"
RUN_STATS = POOL / "run_stats.json"
STAGES = ("broken_timeout", "broken_compiles", "no_located_error", "missing_file", "worker_error", "kept")
mt.LOG = common.PROJ / "out" / "mine_pool.log"
log = mt.log
MAX_CANDIDATES = 8000
MISSING_FILE_RE = re.compile(  # missing files/fonts/executables: a compile-only reward would learn to delete the line
    r"not found|not existent|not installed|not loadable|not available|cannot be found|cannot be opened|could not be opened|"
    r"Cannot find file|Cannot determine size|Cannot be run with|can't find file|can't write on file|can't open file|"
    r"did not find|system call|shell escape|write18|Unknown graphics extension|Package fontspec Error|pdftex|XeTeX|XeLaTeX|"
    r"LuaTeX|Lua\(HB\)|LuaLaTeX|Unicode-compliant|Backend request|auto-pst-pdf|gnuplot-result|externalization failed|Perl module", re.I)


def mwe_hash(text: str) -> str:
    return hashlib.sha1("\n".join(mt.wskey(l) for l in text.split("\n") if l.strip()).encode()).hexdigest()


def parse_questions() -> tuple[list[dict], Counter]:
    stats: Counter = Counter()
    cands: list[dict] = []
    t0 = time.monotonic()
    for a in mt.iter_rows():
        if a.get("PostTypeId") != "1":
            continue
        stats["questions"] += 1
        if stats["questions"] % 100000 == 0:
            log(f"  parsed {stats['questions']} questions, {len(cands)} candidates, {time.monotonic() - t0:.0f}s")
        q = mt.question_candidate(a, stats)
        if q is not None:
            cands.append({**q, "has_accepted": bool(a.get("AcceptedAnswerId"))})
    return cands, stats


def process(c: dict) -> dict:
    qid, broken = c["question_id"], c["broken"]
    d = mt.materialize(qid, broken, broken, base=POOL)
    rb = common.compile_project(d, [], timeout=60)
    stage = "kept"
    err_lines = [] if rb["timed_out"] else common.error_lines_from_items(rb["items"], "main.tex", {"main.tex"})
    if rb["timed_out"]:
        stage = "broken_timeout"
    elif rb["ok"]:
        stage = "broken_compiles"
    elif err_lines[0]["file"] != "main.tex":  # the first error is what the row reports
        stage = "no_located_error"
    elif MISSING_FILE_RE.search(err_lines[0]["text"]):
        stage = "missing_file"
    if stage != "kept":
        shutil.rmtree(d.parent, ignore_errors=True)
        return {"stage": stage}
    prompt = common.build_fix_prompt("main.tex", err_lines, rb["log"], broken,
                                     full_lines=V2_FULL_LINES, max_chars=V2_MAX_CHARS)
    row = {
        "id": f"texse_pool_{qid}", "doc": f"texse_pool/{qid}", "src_dir": str(d), "bundles": [],
        "mutation": "real_unlabeled", "seed": 0, "engine": rb["engine"],
        "compile_seconds": rb["seconds"], "error_lines": err_lines,
        "log_tail": rb["log"][-common.MAX_LOG_CHARS:], "prompt": prompt,
        **{k: c[k] for k in ("question_id", "has_accepted", "title", "tags", "score", "created", "error_message")},
        "license": mt.license_of(c["created"]), "compiled_error": err_lines[0]["text"],
        "error_category": mt.normalize_error(err_lines[0]["text"]),
        "url": f"https://tex.stackexchange.com/q/{qid}",
    }
    return {"stage": "kept", "row": row}


def write_report(rows: list[dict], stats: Counter, path: Path) -> None:
    order = ["questions", "q_error_line", "q_mwe", "q_too_long", "parsed", "in_eval", "external_files",
             "dedup", "candidates", "capped", "processed", "broken_timeout", "broken_compiles",
             "no_located_error", "missing_file", "worker_error", "kept"]
    funnel = "\n".join(f"| {k} | {stats[k]} |" for k in order if k in stats)
    errs = "\n".join(f"| {k} | {v} |" for k, v in Counter(r["error_category"] for r in rows).most_common(20))
    years = dict(sorted(Counter(r["created"][:4] for r in rows).items()))
    acc = Counter(r["has_accepted"] for r in rows)
    path.write_text(f"""# TeX.SE unlabeled broken-doc pool

{len(rows)} rows in `data/texse_pool.jsonl` (CC BY-SA 2.5/3.0/4.0 by post date, `license` per row; attribution via `question_id`/`url`): askers'
MWEs that fail locally with a located error in `main.tex`; no fix labels (compile-reward RL only).
Disjoint from `data/texse_eval.jsonl` by question id and by normalized-MWE hash. Rows whose first
error is a missing file/font/graphic are dropped (a compile-only reward would learn to delete lines).

## Funnel

| stage | count |
|---|---|
{funnel}

has_accepted: {acc[True]} yes / {acc[False]} no. Engines: {dict(Counter(r['engine'] for r in rows))}.
Years: {years}.

## First error (locally compiled, normalized), top 20

| error | rows |
|---|---|
{errs}
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max candidates to compile (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(common.DATA / "texse_pool.jsonl"))
    ap.add_argument("--resume", action="store_true", help="skip qids already in --out")
    args = ap.parse_args()
    mt.LOG.parent.mkdir(parents=True, exist_ok=True)
    POOL.mkdir(parents=True, exist_ok=True)

    if CANDIDATES.exists():
        cands = [json.loads(l) for l in CANDIDATES.read_text().splitlines()]
        stats = Counter(json.loads(PARSE_STATS.read_text()))
        log(f"loaded {len(cands)} cached candidates")
    else:
        mt.ensure_posts()
        log("parsing Posts.xml (questions only)")
        cands, stats = parse_questions()
        with CANDIDATES.open("w") as fh:
            for c in cands:
                fh.write(json.dumps(c) + "\n")
        PARSE_STATS.write_text(json.dumps(stats))
        log(f"parse done: {dict(stats)}")
    stats["parsed"] = len(cands)

    eval_rows = [json.loads(l) for l in (common.DATA / "texse_eval.jsonl").read_text().splitlines()]
    used = {r["question_id"] for r in eval_rows}
    seen = {mwe_hash(mt.run_broken(r)) for r in eval_rows}
    uniq = []
    for c in cands:
        if c["question_id"] in used:
            stats["in_eval"] += 1
        elif mt.needs_external(c["broken"]):
            stats["external_files"] += 1
        elif (h := mwe_hash(c["broken"])) in seen:
            stats["dedup"] += 1
        else:
            seen.add(h)
            uniq.append(c)
    stats["candidates"] = len(uniq)
    if len(uniq) > MAX_CANDIDATES:
        uniq = sorted(uniq, key=lambda c: -c["score"])[:MAX_CANDIDATES]
        stats["capped"] = len(uniq)
    random.Random(args.seed).shuffle(uniq)
    if args.limit:
        uniq = uniq[: args.limit]
    out = Path(args.out)
    rows, done_ids = [], set()
    if args.resume and out.exists():
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        prev = json.loads(RUN_STATS.read_text()) if RUN_STATS.exists() else {"stats": {}, "done": []}
        stats.update({k: v for k, v in prev["stats"].items() if k in STAGES})
        done_ids = {r["question_id"] for r in rows} | set(prev["done"])
        uniq = [c for c in uniq if c["question_id"] not in done_ids]
        log(f"resume: {len(rows)} rows already in {out}, {len(done_ids)} qids done")
    log(f"{len(uniq)} candidates -> compiling with {args.workers} workers")

    t0, done = time.monotonic(), 0
    with out.open("a" if args.resume else "w") as fh, ProcessPoolExecutor(args.workers) as ex:
        pending = {ex.submit(process, c): c for c in uniq[: args.workers * 2]}
        next_i = len(pending)
        while pending:
            fut = next(as_completed(pending))
            done_ids.add(pending.pop(fut)["question_id"])
            done += 1
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = {"stage": "worker_error"}
                log(f"  worker error: {str(e)[:120]}")
            stats[res["stage"]] += 1
            if res["stage"] == "kept":
                rows.append(res["row"])
                fh.write(json.dumps(res["row"]) + "\n")
                fh.flush()
            if done % 50 == 0:
                RUN_STATS.write_text(json.dumps({"stats": stats, "done": sorted(done_ids)}))
                rate = done / (time.monotonic() - t0)
                log(f"  {done}/{len(uniq)} compiled, {len(rows)} kept, ~{(len(uniq) - done) / rate / 60:.0f}m left; "
                    f"stages: {dict((k, stats[k]) for k in STAGES)}")
            if next_i < len(uniq):
                pending[ex.submit(process, uniq[next_i])] = uniq[next_i]
                next_i += 1
    stats["processed"] = len(done_ids)
    RUN_STATS.write_text(json.dumps({"stats": stats, "done": sorted(done_ids)}))
    log(f"done: {len(rows)} rows in {out} ({(time.monotonic() - t0) / 60:.1f}m); stages: {dict(stats)}")
    write_report(rows, stats, out.with_name(out.stem + "_report.md"))
    for k, v in Counter(r["error_category"] for r in rows).most_common(20):
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
