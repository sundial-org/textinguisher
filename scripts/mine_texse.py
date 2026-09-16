"""Mine real LaTeX compile-error fixes from the TeX.StackExchange data dump (CC BY-SA; the version depends on
the post date, see license_of; rows carry it as `license`).

Question with an accepted answer + a pasted `! ...` error + a compilable MWE  ->  BROKEN doc.
Accepted answer's full document (or a snippet spliced into the MWE)             ->  FIXED doc.
Both are compiled locally; break_edits are derived from the line diff so run_eval.py works unchanged.

  python3 mine_texse.py [--target 400] [--workers 8]
Outputs: data/texse_eval.jsonl, data/texse_report.md, corpus/texse/<qid>/files/main.tex.
"""
import argparse
import difflib
import hashlib
import html
import json
import random
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import common
from gen_arxiv import GRAPHICS_RE, PDF_BYTES, PNG_BYTES
from gen_data import V2_FULL_LINES, V2_MAX_CHARS, sites_visible

TEXSE = common.PROJ / "corpus" / "texse"
ARCHIVE = TEXSE / "tex.stackexchange.com.7z"
POSTS = TEXSE / "Posts.xml"
CANDIDATES = TEXSE / "candidates.jsonl"
STATS = TEXSE / "mine_stats.json"
LOG = common.PROJ / "out" / "mine_texse.log"

CODE_RE = re.compile(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
ERR_PREFIXES = (
    "LaTeX Error:", "LaTeX3 Error:", "Package ", "Class ", "Undefined control sequence",
    "Missing ", "Runaway argument", "Extra ", "Misplaced ", "Paragraph ended before",
    "File ended while scanning", "Too many }", "Illegal ", "You can't use", "Argument of",
    "Double su", "Emergency stop", "Use of ", "Not in outer par mode", "Display math should end",
    "Bad ", "I can't find file", "Font ", "Dimension too large", "TeX capacity exceeded",
    "Incomplete ", "Text line contains an invalid character", "Undefined tab position",
    "Improper ", "Unbalanced ", "Command ", "Environment ", "Lonely ", "There's no line here",
    "Something's wrong", "Unknown ", "Cannot ", "Please ", "Limit controls", "Math ",
    "Interruption", "Infinite glue", "Arithmetic overflow", "Number too big",
)
ERR_LINE_RE = re.compile(r"(?m)^[ \t]*(?:! |\S+\.tex:\d+: )(.{5,}?)\s*$")
ERR_MSG_RE = re.compile("^(?:" + "|".join(re.escape(p) for p in ERR_PREFIXES) + ")")
EXTERNAL_RE = re.compile(
    r"\\(?:input|include|addbibresource|lstinputlisting|includepdf|import|subfile|"
    r"verbatiminput|includesvg|includestandalone|InputIfFileExists)\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}")
FILECONTENTS_RE = re.compile(r"\\begin\{filecontents\*?\}\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}")
MAX_LINES, MAX_CHARS = 400, 14000
MAX_HUNKS, MAX_DIFF_LINES = 6, 40


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


# --- phase A: stream Posts.xml ---

def norm_code(raw: str) -> str:
    text = html.unescape(raw).replace("\r\n", "\n").replace("\r", "\n")
    lines = [l.rstrip() for l in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n"


def code_blocks(body: str) -> list[str]:
    return [norm_code(m.group(1)) for m in CODE_RE.finditer(body)]


def is_full_doc(code: str) -> bool:
    return "\\documentclass" in code and "\\begin{document}" in code


def error_messages(body: str) -> list[str]:
    text = html.unescape(TAG_RE.sub("\n", body))
    return [m.group(1) for m in ERR_LINE_RE.finditer(text) if ERR_MSG_RE.match(m.group(1))]


def ensure_posts() -> None:
    if POSTS.exists():
        return
    log(f"extracting Posts.xml from {ARCHIVE.name}")
    subprocess.run(["7zz", "e", "-y", str(ARCHIVE), "Posts.xml", f"-o{TEXSE}"],
                   check=True, capture_output=True)
    log(f"extracted Posts.xml ({POSTS.stat().st_size / 1e9:.2f} GB)")


def iter_rows():
    for _, el in ET.iterparse(str(POSTS), events=("end",)):
        if el.tag == "row":
            yield el.attrib
        el.clear()


def question_candidate(a: dict, stats: Counter) -> dict | None:
    """Question row -> metadata + broken MWE if it pastes an error and a full document within limits."""
    body = a.get("Body", "")
    errs = error_messages(body)
    if not errs:
        return None
    stats["q_error_line"] += 1
    docs = [c for c in code_blocks(body) if is_full_doc(c)]
    if not docs:
        return None
    stats["q_mwe"] += 1
    broken = max(docs, key=len)
    if broken.count("\n") > MAX_LINES or len(broken) > MAX_CHARS:
        stats["q_too_long"] += 1
        return None
    return {
        "question_id": int(a["Id"]), "title": html.unescape(a.get("Title", "")),
        "tags": re.findall(r"<([^<>]+)>", a.get("Tags", "")) or
                [t for t in a.get("Tags", "").strip("|").split("|") if t],
        "score": int(a.get("Score", 0)), "created": a.get("CreationDate", "")[:10],
        "error_message": errs[0], "broken": broken,
    }


def parse_posts() -> tuple[list[dict], Counter]:
    """Two passes (rows are not sorted by Id): candidate questions, then their accepted answers."""
    stats: Counter = Counter()
    waiting: dict[int, dict] = {}  # accepted answer id -> question
    t0 = time.monotonic()
    for a in iter_rows():
        if a.get("PostTypeId") != "1":
            continue
        stats["questions"] += 1
        if stats["questions"] % 100000 == 0:
            log(f"  parsed {stats['questions']} questions, {len(waiting)} candidates, "
                f"{time.monotonic() - t0:.0f}s")
        acc = a.get("AcceptedAnswerId")
        if not acc:
            continue
        stats["q_accepted"] += 1
        q = question_candidate(a, stats)
        if q is not None:
            waiting[int(acc)] = {**q, "accepted_answer_id": int(acc)}
    log(f"  pass 1 done: {len(waiting)} candidate questions; scanning answers")
    cands: list[dict] = []
    for a in iter_rows():
        if a.get("PostTypeId") != "2":
            continue
        q = waiting.pop(int(a["Id"]), None)
        if q is None:
            continue
        q["answer_blocks"] = code_blocks(a.get("Body", ""))
        if q["answer_blocks"]:
            cands.append(q)
            stats["candidates"] += 1
        else:
            stats["answer_no_code"] += 1
    stats["answer_missing"] = len(waiting)
    return cands, stats


# --- fixed-doc derivation ---

def wskey(line: str) -> str:
    return " ".join(line.split())


def indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def reindent(seg: list[str], ref: list[str], prev: str) -> list[str]:
    """Give the answer's changed lines the broken doc's indentation (line-wise when the
    hunk is a same-size replace, else a uniform shift), so diffs carry content only."""
    nb_seg = [l for l in seg if l.strip()]
    nb_ref = [l for l in ref if l.strip()]
    if len(seg) == len(ref) and all(bool(a.strip()) == bool(b.strip()) for a, b in zip(seg, ref)):
        return [indent_of(b) + a.lstrip() if a.strip() else a for a, b in zip(seg, ref)]
    if not nb_seg:
        return seg
    src = min((indent_of(l) for l in nb_seg), key=len)
    dst = min((indent_of(l) for l in nb_ref), key=len) if nb_ref else indent_of(prev)
    return [dst + l[len(src):] if l.startswith(src) and l.strip() else l for l in seg]


def merge_whitespace(broken: str, answer: str) -> str:
    """Rebuild the fixed doc from the broken doc's lines wherever they match the answer's
    modulo whitespace; blank lines and indentation always come from the broken doc, so the
    diff carries content changes only."""
    bl = broken.rstrip("\n").split("\n")
    al = [l for l in answer.split("\n") if l.strip()]
    nb = [i for i, l in enumerate(bl) if l.strip()]

    def own(k: int) -> list[str]:  # content line k plus the blank lines trailing it
        return bl[nb[k]: nb[k + 1] if k + 1 < len(nb) else len(bl)]

    sm = difflib.SequenceMatcher(None, [wskey(bl[i]) for i in nb], [wskey(l) for l in al],
                                 autojunk=False)
    out = bl[: nb[0]] if nb else []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                out.extend(own(k))
            continue
        out.extend(reindent(al[j1:j2], [bl[i] for i in nb[i1:i2]], bl[nb[i1 - 1]] if i1 else ""))
        if i2 > i1:
            out.extend(own(i2 - 1)[1:])
    return "\n".join(out) + "\n"


def splice_snippet(broken: str, snippet: str) -> str | None:
    """Replace the MWE region located by the snippet's uniquely-matching lines (edge lines may
    be the changed ones); a lone line falls back to its closest unique match."""
    lines = broken.split("\n")
    bl = [wskey(l) for l in lines]
    sl = [l for l in snippet.split("\n") if l.strip()]
    if not sl:
        return None
    anchors = []
    for k, l in enumerate(sl):
        hits = [i for i, b in enumerate(bl) if b == wskey(l)]
        if len(hits) == 1 and (not anchors or hits[0] > anchors[-1][1]):
            anchors.append((k, hits[0]))
    if anchors:
        start = anchors[0][1] - anchors[0][0]
        end = anchors[-1][1] + (len(sl) - 1 - anchors[-1][0]) + 1
    elif len(sl) == 1:
        scored = sorted(((difflib.SequenceMatcher(None, wskey(sl[0]), b).ratio(), i)
                         for i, b in enumerate(bl) if b), reverse=True)
        if not scored or scored[0][0] < 0.6 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
            return None
        start, end = scored[0][1], scored[0][1] + 1
    else:
        return None
    if start < 0 or end > len(lines) or end < start:
        return None
    return "\n".join(lines[:start] + sl + lines[end:])


def fix_candidates(c: dict) -> list[tuple[str, str]]:
    """(source, fixed_text) candidates in preference order, cheap filters only."""
    broken = c["broken"]
    out, seen = [], set()
    full = sorted((b for b in c["answer_blocks"] if is_full_doc(b)), key=len, reverse=True)
    for b in full:
        out.append(("answer_doc", merge_whitespace(broken, b)))
    if not full:
        for b in c["answer_blocks"]:
            spliced = splice_snippet(broken, b)
            if spliced:
                out.append(("snippet", merge_whitespace(broken, spliced)))
    uniq = []
    for src, fixed in out:
        if fixed != broken and fixed not in seen:
            seen.add(fixed)
            uniq.append((src, fixed))
    return uniq[:3]


def derive_edits(fixed: str, broken: str) -> tuple[list[dict] | None, str]:
    fl, bl = fixed.rstrip("\n").split("\n"), broken.rstrip("\n").split("\n")
    sm = difflib.SequenceMatcher(None, fl, bl, autojunk=False)
    hunks: list[list[int]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if hunks and i1 - hunks[-1][1] <= 1:
            hunks[-1][1], hunks[-1][3] = i2, j2
        else:
            hunks.append([i1, i2, j1, j2])
    n_diff = sum((i2 - i1) + (j2 - j1) for i1, i2, j1, j2 in hunks)
    if not hunks:
        return None, "identical"
    if len(hunks) > MAX_HUNKS or n_diff > MAX_DIFF_LINES:
        return None, "diff_too_big"
    edits = []
    for i1, i2, j1, j2 in hunks:
        found = None
        for ctx in range(0 if (i2 > i1 and j2 > j1) else 1, 6):
            s = "\n".join(fl[max(0, i1 - ctx):i2 + ctx])
            r = "\n".join(bl[max(0, j1 - ctx):j2 + ctx])
            if s.strip() and r.strip() and s.split("\n")[-1].strip() and r.split("\n")[-1].strip() \
                    and fixed.count(s) == 1 and broken.count(r) == 1:  # edit blocks cannot end with a blank line
                found = {"type": "real", "search": s, "replace": r}
                break
        if not found:
            return None, "edit_not_unique"
        edits.append(found)
    # exact round trip, both directions, with the same sequential semantics as run_eval
    doc = fixed
    for e in edits:
        if doc.count(e["search"]) != 1:
            return None, "edit_not_unique"
        doc = doc.replace(e["search"], e["replace"], 1)
    if doc != broken:
        return None, "roundtrip_mismatch"
    target = common.format_edits([(e["replace"], e["search"]) for e in reversed(edits)])
    back, err = common.apply_edits(broken, common.parse_edits(target))
    if err or back != fixed:
        return None, "roundtrip_mismatch"
    return edits, "ok"


def needs_external(text: str) -> bool:
    provided = {Path(n.strip()).name for n in FILECONTENTS_RE.findall(text)}
    for ref in EXTERNAL_RE.findall(text):
        name = Path(ref.strip()).name
        if name not in provided and name + ".tex" not in provided:
            return True
    return False


def license_of(created: str) -> str:
    """Stack Exchange content licence by post date: CC BY-SA 2.5 before 2011-04-08, 3.0 until 2018-05-02, 4.0 after."""
    return "CC-BY-SA-" + ("2.5" if created < "2011-04-08" else "3.0" if created < "2018-05-02" else "4.0")


def materialize(qid: int, fixed: str, broken: str, base: Path = TEXSE) -> Path:
    d = base / str(qid) / "files"
    if d.exists():
        shutil.rmtree(d.parent)
    d.mkdir(parents=True)
    (d / "main.tex").write_text(fixed)
    for m in GRAPHICS_RE.finditer(fixed + broken):
        ref = Path(m.group(1).strip()).name
        if ref.startswith("example-"):
            continue
        target = d / (ref if "." in ref else ref + ".pdf")
        if not target.exists():
            from gen_skew import placeholder  # PNG_BYTES is rejected by pdftex's libpng: only image-deleting hacks compiled
            target.write_bytes(placeholder(target) if target.suffix.lower() in (".png", ".jpg", ".jpeg") else PDF_BYTES)
    return d


def normalize_error(msg: str) -> str:
    msg = re.sub(r"^(?:! |\S+\.tex:\d+: )", "", msg)
    msg = re.sub(r"`[^']*'|'[^']*'|\"[^\"]*\"", "'X'", msg)
    msg = re.sub(r"\\[A-Za-z@]+", r"\\CS", msg)
    msg = re.sub(r"\{[^{}]*\}", "{X}", msg)
    msg = re.sub(r"\d+", "N", msg)
    return msg[:70].strip()


# --- phase B: compile + build row (worker) ---

def process(c: dict) -> dict:
    qid, broken = c["question_id"], c["broken"]
    if needs_external(broken):
        return {"stage": "external_files"}
    fixes = fix_candidates(c)
    if not fixes:
        return {"stage": "no_fix_candidate"}
    plans, reasons = [], Counter()
    for src, fixed in fixes:
        if needs_external(fixed):
            reasons["external_files"] += 1
            continue
        edits, why = derive_edits(fixed, broken)
        if edits:
            plans.append((src, fixed, edits))
        else:
            reasons[why] += 1
    if not plans:
        return {"stage": reasons.most_common(1)[0][0]}
    d = materialize(qid, plans[0][1], broken)
    res = _compile_plans(c, d, plans)
    if res["stage"] != "kept":
        shutil.rmtree(d.parent, ignore_errors=True)
    return res


def _compile_plans(c: dict, d: Path, plans: list) -> dict:
    qid, broken = c["question_id"], c["broken"]
    rb = common.compile_project(d, [], overrides={"main.tex": broken}, timeout=60)
    if rb["timed_out"]:
        return {"stage": "broken_timeout"}
    if rb["ok"]:
        return {"stage": "broken_compiles"}
    err_lines = common.error_lines_from_items(rb["items"], "main.tex", {"main.tex"})
    if not err_lines:
        return {"stage": "broken_no_located_error"}
    if re.search(r"File .* not found", err_lines[0]["text"]) and "not found" not in c["error_message"]:
        return {"stage": "local_file_missing"}  # asker's private .sty/.cls, not the reported error
    prompt = common.build_fix_prompt("main.tex", err_lines, rb["log"], broken,
                                     full_lines=V2_FULL_LINES, max_chars=V2_MAX_CHARS)
    last = "fixed_compile_fail"
    for src, fixed, edits in plans:
        if not sites_visible(prompt, edits):
            last = "sites_hidden"
            continue
        (d / "main.tex").write_text(fixed)
        rf = common.compile_project(d, [], timeout=60)
        if rf["timed_out"]:
            last = "fixed_timeout"
            continue
        if not rf["ok"]:
            last = "fixed_compile_fail"
            continue
        target = common.format_edits([(e["replace"], e["search"]) for e in reversed(edits)])
        row = {
            "id": f"texse_{qid}", "doc": f"texse/{qid}", "src_dir": str(d), "bundles": [],
            "mutation": "real", "seed": 0, "break_edits": edits, "n_breaks": len(edits),
            "engine": rb["engine"], "compile_seconds": rb["seconds"],
            "error_lines": err_lines, "log_tail": rb["log"][-common.MAX_LOG_CHARS:],
            "prompt": prompt, "target": target,
            "question_id": qid, "accepted_answer_id": c["accepted_answer_id"],
            "title": c["title"], "tags": c["tags"], "score": c["score"],
            "created": c["created"], "error_message": c["error_message"], "license": license_of(c["created"]),
            "compiled_error": err_lines[0]["text"],
            "error_category": normalize_error(err_lines[0]["text"]),
            "fix_source": src, "diff_lines": sum(
                len(e["search"].split("\n")) + len(e["replace"].split("\n")) for e in edits),
            "url": f"https://tex.stackexchange.com/q/{qid}",
        }
        return {"stage": "kept", "row": row}
    return {"stage": last}


# --- report ---

def write_report(rows: list[dict], stats: Counter, path: Path) -> None:
    def hist(counter: Counter, n: int) -> str:
        return "\n".join(f"| {k} | {v} |" for k, v in counter.most_common(n))

    errs = Counter(r["error_category"] for r in rows)
    tags = Counter(t for r in rows for t in r["tags"])
    q_errs = Counter(normalize_error(r["error_message"]) for r in rows)
    match = sum(normalize_error(r["error_message"]).split("'")[0][:25] ==
                r["error_category"].split("'")[0][:25] for r in rows)
    src = Counter(r["fix_source"] for r in rows)
    years = Counter(r["created"][:4] for r in rows)
    stage_order = ["questions", "q_accepted", "q_error_line", "q_mwe", "q_too_long",
                   "answer_missing", "answer_no_code", "candidates", "processed", "dedup",
                   "external_files", "no_fix_candidate", "identical", "diff_too_big",
                   "edit_not_unique", "roundtrip_mismatch", "broken_timeout", "broken_compiles",
                   "broken_no_located_error", "local_file_missing", "sites_hidden", "fixed_timeout",
                   "fixed_compile_fail", "kept"]
    funnel = "\n".join(f"| {k} | {stats[k]} |" for k in stage_order if k in stats)
    examples = []
    rng = random.Random(3)
    for r in rng.sample(rows, min(3, len(rows))):
        diff = "\n".join(difflib.unified_diff(
            run_broken(r).split("\n"), Path(r["src_dir"], "main.tex").read_text().split("\n"),
            "broken/main.tex", "fixed/main.tex", lineterm="", n=1))
        examples.append(f"### [{r['title']}]({r['url']}) (q{r['question_id']}, tags: "
                        f"{', '.join(r['tags'])})\n\nError: `{r['compiled_error']}`\n\n"
                        f"```diff\n{diff}\n```")
    n_edits = Counter(r["n_breaks"] for r in rows)
    engines = Counter(r["engine"] for r in rows)
    path.write_text(f"""# TeX.SE real-world compile-error eval

{len(rows)} rows in `data/texse_eval.jsonl`, mined from the TeX.StackExchange data dump
(archive.org `stackexchange_20251231`). Each row: the asker's MWE (fails locally with a located
error) + the accepted answer's document (compiles clean locally), diffed into `break_edits` so
`run_eval.py` works unchanged. Content is CC BY-SA 2.5/3.0/4.0 by post date (`license` per row; attribution via `question_id`/`url`).

## Funnel

| stage | count |
|---|---|
{funnel}

Fix source: {dict(src)}. Engines: {dict(engines)}. Edits per row: {dict(sorted(n_edits.items()))}.
Question-reported error category matches the locally compiled one in {match}/{len(rows)} rows.
Years: {dict(sorted(years.items()))}.

## Error distribution (locally compiled first error, normalized)

| error | rows |
|---|---|
{hist(errs, 30)}

## Error reported in the question (normalized)

| error | rows |
|---|---|
{hist(q_errs, 20)}

## Tags

| tag | rows |
|---|---|
{hist(tags, 30)}

## Examples

{chr(10).join(examples)}
""")


def run_broken(row: dict) -> str:
    doc = Path(row["src_dir"], "main.tex").read_text()
    for e in row["break_edits"]:
        doc = doc.replace(e["search"], e["replace"], 1)
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="max candidates to compile")
    ap.add_argument("--out", default=str(common.DATA / "texse_eval.jsonl"))
    ap.add_argument("--append", action="store_true", help="append to --out, accumulate stats")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N shuffled candidates")
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)

    if CANDIDATES.exists():
        cands = [json.loads(l) for l in CANDIDATES.read_text().splitlines()]
        stats = Counter(json.loads((TEXSE / "parse_stats.json").read_text()))
        log(f"loaded {len(cands)} cached candidates")
    else:
        ensure_posts()
        log("parsing Posts.xml")
        cands, stats = parse_posts()
        with CANDIDATES.open("w") as fh:
            for c in cands:
                fh.write(json.dumps(c) + "\n")
        (TEXSE / "parse_stats.json").write_text(json.dumps(stats))
        log(f"parse done: {dict(stats)}")

    seen = set()
    uniq = []
    for c in cands:
        h = hashlib.sha1(c["broken"].encode()).hexdigest()
        if h in seen:
            stats["dedup"] += 1
        else:
            seen.add(h)
            uniq.append(c)
    random.Random(args.seed).shuffle(uniq)
    uniq = uniq[args.skip: args.skip + args.limit if args.limit else None]
    log(f"{len(uniq)} unique candidates -> compiling with {args.workers} workers, target {args.target}")

    rows, t0 = [], time.monotonic()
    out = Path(args.out)
    if args.append:
        stats = Counter(json.loads(STATS.read_text())) if STATS.exists() else stats
        old = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
        done_ids = {r["question_id"] for r in old}
        uniq = [c for c in uniq if c["question_id"] not in done_ids]
    with out.open("a" if args.append else "w") as fh, ProcessPoolExecutor(args.workers) as ex:
        pending = {ex.submit(process, c): c for c in uniq[: args.workers * 2]}
        next_i = len(pending)
        done = 0
        while pending:
            fut = next(as_completed(pending))
            pending.pop(fut)
            done += 1
            stats["processed"] += 1
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
            if done % 25 == 0:
                rate = done / (time.monotonic() - t0)
                left = min(len(uniq) - done, max(0, args.target - len(rows)) / max(len(rows), 1) * done)
                log(f"  {done}/{len(uniq)} processed, {len(rows)} kept "
                    f"({len(rows) / done:.0%}), ~{left / rate / 60:.0f}m left")
            if len(rows) < args.target and next_i < len(uniq):
                pending[ex.submit(process, uniq[next_i])] = uniq[next_i]
                next_i += 1
    log(f"done: {len(rows)} rows in {out} ({(time.monotonic() - t0) / 60:.1f}m); "
        f"stages: {dict(stats)}")
    STATS.write_text(json.dumps(stats))
    if args.append:
        rows = old + rows
    write_report(rows, stats, common.DATA / "texse_report.md")
    print("\nerror histogram:")
    for k, v in Counter(r["error_category"] for r in rows).most_common(25):
        print(f"  {v:4d}  {k}")
    print("\ntag histogram:")
    for k, v in Counter(t for r in rows for t in r["tags"]).most_common(25):
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
