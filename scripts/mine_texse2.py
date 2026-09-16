"""Relaxed TeX.SE miner: any question whose title/body MENTIONS an error (prose, log paste, or
`! ...` line) and contains a full-document MWE. Fix source: accepted answer, else the top-scoring
answer (score >= 2). Same compile funnel as mine_texse.py; broken-but-unfixed docs go to the pool.

  ../.venv/bin/python -u scripts/mine_texse2.py [--limit N] [--workers 6] [--resume]
Outputs: data/texse2_labeled.jsonl, data/texse2_pool.jsonl, corpus/texse2/<qid>/files/main.tex.
"""
import argparse
import html
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
from gen_data import V2_FULL_LINES, V2_MAX_CHARS, sites_visible
from mine_texse_pool import MISSING_FILE_RE, mwe_hash

TEXSE2 = common.PROJ / "corpus" / "texse2"
CANDIDATES = TEXSE2 / "candidates.jsonl"
PARSE_STATS = TEXSE2 / "parse_stats.json"
RUN_STATS = TEXSE2 / "run_stats.json"
mt.LOG = common.PROJ / "out" / "mine_texse2.log"
log = mt.log
MAX_CANDIDATES = 40000
STAGES = ("broken_timeout", "broken_compiles", "no_located_error", "missing_file", "worker_error", "kept", "pool")
FIX_STAGES = ("no_fix_candidate", "external_files", "identical", "diff_too_big", "edit_not_unique",
              "roundtrip_mismatch", "sites_hidden", "fixed_timeout", "fixed_compile_fail")
MISSING_RE = re.compile(MISSING_FILE_RE.pattern + r"|Could not read|No file", re.I)
ERR_MENTION_RE = re.compile(
    r"error|undefined control sequence|missing|runaway|extra \}|misplaced|does ?n[o']t compile|"
    r"won'?t compile|fails? to compile|emergency stop|^[ \t]*! |\bl\.\d+", re.I | re.M)


def parse_posts() -> tuple[list[dict], Counter]:
    """Pass 1: questions mentioning an error with a full-doc MWE. Pass 2: their accepted and
    top-scoring (>= 2) answers' code blocks."""
    stats: Counter = Counter()
    cands: dict[int, dict] = {}
    t0 = time.monotonic()
    for a in mt.iter_rows():
        if a.get("PostTypeId") != "1":
            continue
        stats["questions"] += 1
        if stats["questions"] % 100000 == 0:
            log(f"  parsed {stats['questions']} questions, {len(cands)} candidates, {time.monotonic() - t0:.0f}s")
        body = a.get("Body", "")
        docs = [c for c in mt.code_blocks(body) if mt.is_full_doc(c)]
        if not docs:
            continue
        stats["q_mwe"] += 1
        text = a.get("Title", "") + "\n" + html.unescape(mt.TAG_RE.sub("\n", body))
        m = ERR_MENTION_RE.search(text)
        if not m:
            continue
        stats["q_error_mention"] += 1
        broken = max(docs, key=len)
        if broken.count("\n") > mt.MAX_LINES or len(broken) > mt.MAX_CHARS:
            stats["q_too_long"] += 1
            continue
        errs = mt.error_messages(body)
        cands[int(a["Id"])] = {
            "question_id": int(a["Id"]), "title": html.unescape(a.get("Title", "")),
            "tags": re.findall(r"<([^<>]+)>", a.get("Tags", "")) or
                    [t for t in a.get("Tags", "").strip("|").split("|") if t],
            "score": int(a.get("Score", 0)), "created": a.get("CreationDate", "")[:10],
            "error_message": errs[0] if errs else "", "error_mention": m.group(0).strip().lower(),
            "broken": broken, "accepted_answer_id": int(a["AcceptedAnswerId"]) if a.get("AcceptedAnswerId") else None,
            "accepted_blocks": [], "top_blocks": [], "top_answer_id": None, "top_score": 0, "accepted_score": 0,
        }
    log(f"  pass 1 done: {len(cands)} candidate questions; scanning answers")
    for a in mt.iter_rows():
        if a.get("PostTypeId") != "2":
            continue
        q = cands.get(int(a.get("ParentId", 0)))
        if q is None:
            continue
        score, aid = int(a.get("Score", 0)), int(a["Id"])
        blocks = mt.code_blocks(a.get("Body", ""))
        if aid == q["accepted_answer_id"]:
            q["accepted_blocks"], q["accepted_score"] = blocks, score
        elif score >= 2 and score > q["top_score"] and blocks:
            q["top_blocks"], q["top_answer_id"], q["top_score"] = blocks, aid, score
    out = []
    for q in cands.values():
        stats["has_accepted" if q["accepted_blocks"] else ("has_top" if q["top_blocks"] else "no_answer_code")] += 1
        out.append(q)
    stats["candidates"] = len(out)
    return out, stats


def process(c: dict) -> dict:
    qid, broken = c["question_id"], c["broken"]
    plans, reasons = [], Counter()
    for kind, blocks, aid, ascore in (("accepted", c["accepted_blocks"], c["accepted_answer_id"], c["accepted_score"]),
                                      ("top", c["top_blocks"], c["top_answer_id"], c["top_score"])):
        for src, fixed in mt.fix_candidates({"broken": broken, "answer_blocks": blocks}):
            if mt.needs_external(fixed):
                reasons["external_files"] += 1
                continue
            edits, why = mt.derive_edits(fixed, broken)
            if edits:
                plans.append((kind, aid, ascore, src, fixed, edits))
            else:
                reasons[why] += 1
    d = mt.materialize(qid, broken, plans[0][4] if plans else "", base=TEXSE2)  # main.tex = broken first; the fix is written over it below
    rb = common.compile_project(d, [], timeout=60)
    err_lines = [] if rb["timed_out"] else common.error_lines_from_items(rb["items"], "main.tex", {"main.tex"})
    stage = "kept"
    if rb["timed_out"]:
        stage = "broken_timeout"
    elif rb["ok"]:
        stage = "broken_compiles"
    elif not err_lines or err_lines[0]["file"] != "main.tex":  # the first error is what the row reports
        stage = "no_located_error"
    elif MISSING_RE.search(err_lines[0]["text"]):
        stage = "missing_file"
    if stage != "kept":
        shutil.rmtree(d.parent, ignore_errors=True)
        return {"stage": stage}
    prompt = common.build_fix_prompt("main.tex", err_lines, rb["log"], broken,
                                     full_lines=V2_FULL_LINES, max_chars=V2_MAX_CHARS)
    row = {
        "id": f"texse2_{qid}", "doc": f"texse2/{qid}", "src_dir": str(d), "bundles": [],
        "mutation": "real", "seed": 0, "engine": rb["engine"], "compile_seconds": rb["seconds"],
        "error_lines": err_lines, "log_tail": rb["log"][-common.MAX_LOG_CHARS:], "prompt": prompt,
        **{k: c[k] for k in ("question_id", "accepted_answer_id", "title", "tags", "score", "created",
                             "error_message", "error_mention")},  # error_message is "" here: `! ` questions went to texse_eval/pool
        "license": mt.license_of(c["created"]),
        "compiled_error": err_lines[0]["text"], "error_category": mt.normalize_error(err_lines[0]["text"]),
        "url": f"https://tex.stackexchange.com/q/{qid}",
    }
    fix_stage = reasons.most_common(1)[0][0] if reasons else "no_fix_candidate"
    for kind, aid, ascore, src, fixed, edits in plans:
        if not sites_visible(prompt, edits):
            fix_stage = "sites_hidden"
            continue
        (d / "main.tex").write_text(fixed)
        rf = common.compile_project(d, [], timeout=60)
        if rf["timed_out"]:
            fix_stage = "fixed_timeout"
            continue
        if not rf["ok"]:
            fix_stage = "fixed_compile_fail"
            continue
        target = common.format_edits([(e["replace"], e["search"]) for e in reversed(edits)])
        row.update(break_edits=edits, n_breaks=len(edits), target=target, fix_source=src, fix_answer=kind,
                   answer_id=aid, answer_score=ascore,
                   diff_lines=sum(len(e["search"].split("\n")) + len(e["replace"].split("\n")) for e in edits))
        return {"stage": "kept", "row": row}
    (d / "main.tex").write_text(broken)
    row.update(id=f"texse2_pool_{qid}", mutation="real_unlabeled", has_accepted=bool(c["accepted_answer_id"]),
               fix_stage=fix_stage)
    return {"stage": "pool", "row": row}


def write_report(labeled: list[dict], pool: list[dict], stats: Counter, path: Path) -> None:
    order = ["questions", "q_mwe", "q_error_mention", "q_too_long", "candidates", "has_accepted", "has_top",
             "no_answer_code", "in_eval_or_pool", "pool_rejected", "external_files", "dedup", "to_compile",
             "capped", "processed", *STAGES]
    funnel = "\n".join(f"| {k} | {stats[k]} |" for k in order if k in stats)
    fixf = {k: stats["fix_" + k] for k in FIX_STAGES if "fix_" + k in stats}
    years = dict(sorted(Counter(r["created"][:4] for r in labeled).items()))
    ans = Counter(f"{r['fix_answer']}/{r['fix_source']}" for r in labeled)
    sc = Counter(("<2" if r["answer_score"] < 2 else "2-4" if r["answer_score"] < 5 else "5-19" if r["answer_score"] < 20
                  else ">=20") for r in labeled)
    mention = Counter(r["error_mention"] for r in labeled)
    errs = Counter(r["error_category"] for r in labeled)
    path.write_text(f"""# TeX.SE relaxed mining (texse2)

{len(labeled)} labeled rows in `data/texse2_labeled.jsonl` (`texse_eval` schema + `fix_answer`, `answer_id`, `answer_score`,
`error_mention`), {len(pool)} unlabeled in `data/texse2_pool.jsonl` (`texse_pool` schema + `fix_stage`). Candidates: title/body
mentions an error (not only a pasted `! ...` line) + full-document MWE; fix from the accepted answer, else the top answer
(score >= 2). Disjoint from `texse_eval`/`texse_pool` by question id and MWE hash. CC BY-SA 2.5/3.0/4.0 by post date (`license` per row; attribution via `url`).

| stage | count |
|---|---|
{funnel}

Why a broken doc landed in the pool (`fix_stage`): {fixf}.
Fix answer/source: {dict(ans)}. Answer score: {dict(sc)}. Engines: {dict(Counter(r['engine'] for r in labeled))}.
Edits per row: {dict(sorted(Counter(r['n_breaks'] for r in labeled).items()))}. Years: {years}.
Error mention that qualified the question (labeled rows): {dict(mention.most_common(8))}.
Pool first errors, top 5: {dict(Counter(r['error_category'] for r in pool).most_common(5))}.

| first error (locally compiled, normalized), labeled rows, top 15 | rows |
|---|---|
{chr(10).join(f"| {k} | {v} |" for k, v in errs.most_common(15))}
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max candidates to compile (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="skip qids already processed")
    args = ap.parse_args()
    mt.LOG.parent.mkdir(parents=True, exist_ok=True)
    TEXSE2.mkdir(parents=True, exist_ok=True)
    labeled_out, pool_out = common.DATA / "texse2_labeled.jsonl", common.DATA / "texse2_pool.jsonl"

    if CANDIDATES.exists():
        cands = [json.loads(l) for l in CANDIDATES.read_text().splitlines()]
        stats = Counter(json.loads(PARSE_STATS.read_text()))
        log(f"loaded {len(cands)} cached candidates")
    else:
        mt.ensure_posts()
        log("parsing Posts.xml")
        cands, stats = parse_posts()
        with CANDIDATES.open("w") as fh:
            for c in cands:
                fh.write(json.dumps(c) + "\n")
        PARSE_STATS.write_text(json.dumps(stats))
        log(f"parse done: {dict(stats)}")

    used, seen = set(), set()
    for name in ("texse_eval.jsonl", "texse_pool.jsonl"):
        for l in (common.DATA / name).read_text().splitlines():
            r = json.loads(l)
            used.add(r["question_id"])
            p = Path(r["src_dir"], "main.tex")
            if p.exists():
                seen.add(mwe_hash(mt.run_broken(r) if "break_edits" in r else p.read_text()))
    pool_stats = common.PROJ / "corpus" / "texse_pool" / "run_stats.json"
    rejected = set(json.loads(pool_stats.read_text())["done"]) - used if pool_stats.exists() else set()
    uniq = []
    for c in cands:
        if c["question_id"] in used:
            stats["in_eval_or_pool"] += 1
        elif c["question_id"] in rejected:
            stats["pool_rejected"] += 1  # broken doc already compiled clean / unlocated / missing file
        elif mt.needs_external(c["broken"]):
            stats["external_files"] += 1
        elif (h := mwe_hash(c["broken"])) in seen:
            stats["dedup"] += 1
        else:
            seen.add(h)
            uniq.append(c)
    stats["to_compile"] = len(uniq)
    if len(uniq) > MAX_CANDIDATES:
        uniq = sorted(uniq, key=lambda c: -c["score"])[:MAX_CANDIDATES]
        stats["capped"] = len(uniq)
    random.Random(args.seed).shuffle(uniq)
    if args.limit:
        uniq = uniq[: args.limit]
    labeled, pool, done_ids = [], [], set()
    if args.resume and RUN_STATS.exists():
        prev = json.loads(RUN_STATS.read_text())
        stats.update({k: v for k, v in prev["stats"].items() if k in STAGES or k.startswith("fix_")})
        labeled = [json.loads(l) for l in labeled_out.read_text().splitlines()] if labeled_out.exists() else []
        pool = [json.loads(l) for l in pool_out.read_text().splitlines()] if pool_out.exists() else []
        done_ids = set(prev["done"]) | {r["question_id"] for r in labeled + pool}
        uniq = [c for c in uniq if c["question_id"] not in done_ids]
        log(f"resume: {len(labeled)} labeled, {len(pool)} pool, {len(done_ids)} qids done")
    log(f"{len(uniq)} candidates -> compiling with {args.workers} workers")

    t0, done = time.monotonic(), 0
    mode = "a" if args.resume else "w"
    with labeled_out.open(mode) as fl, pool_out.open(mode) as fp, ProcessPoolExecutor(args.workers) as ex:
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
                labeled.append(res["row"])
                fl.write(json.dumps(res["row"]) + "\n")
                fl.flush()
            elif res["stage"] == "pool":
                stats["fix_" + res["row"]["fix_stage"]] += 1
                pool.append(res["row"])
                fp.write(json.dumps(res["row"]) + "\n")
                fp.flush()
            if done % 100 == 0:
                RUN_STATS.write_text(json.dumps({"stats": stats, "done": sorted(done_ids)}))
                rate = done / (time.monotonic() - t0)
                log(f"  {done}/{len(uniq)} processed, {len(labeled)} labeled, {len(pool)} pool, "
                    f"~{(len(uniq) - done) / rate / 60:.0f}m left; stages: {dict((k, stats[k]) for k in STAGES)}")
            if next_i < len(uniq):
                pending[ex.submit(process, uniq[next_i])] = uniq[next_i]
                next_i += 1
    stats["processed"] = len(done_ids)
    RUN_STATS.write_text(json.dumps({"stats": stats, "done": sorted(done_ids)}))
    log(f"done: {len(labeled)} labeled, {len(pool)} pool ({(time.monotonic() - t0) / 60:.1f}m); stages: {dict(stats)}")
    write_report(labeled, pool, stats, common.DATA / "texse2_report.md")
    for k, v in Counter(r["error_category"] for r in labeled).most_common(15):
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
