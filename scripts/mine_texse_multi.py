"""TeX.SE questions whose MWE references an external file (\\input/\\include/\\bibliography/\\usepackage) that is pasted in
another code block of the question -> multi-file project rows (asker's project fails with a located error, accepted or top
answer's version compiles). Same funnel as mine_github_multi.check_pair; rows in the project-lane schema.

  mine_texse_multi.py parse                                   # Posts.xml -> corpus/texse_multi/candidates.jsonl
  COMPILE_BACKEND=modal MODAL_PROFILE=sundial-prod mine_texse_multi.py mine [--limit N] [--workers 32]
"""
import argparse
import html
import json
import re
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import common
import mine_github_multi as m1
import mine_texse as mt
from mine_texse2 import ERR_MENTION_RE
from mine_texse_pool import mwe_hash

TM = common.PROJ / "corpus" / "texse_multi"
CANDS, DONE, STATS, REFS = TM / "candidates.jsonl", TM / "done.jsonl", TM / "stats.json", TM / "refs.jsonl"
OUT = common.DATA / "texse_multi.jsonl"
mt.LOG = common.PROJ / "out" / "mine_texse_multi.log"
log = mt.log
REF_RE = re.compile(r"\\(input|include|subfile|bibliography|addbibresource|lstinputlisting|verbatiminput|InputIfFileExists|usepackage|documentclass)"
                    r"\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}")
NAME_RE = re.compile(r"([\w./-]+\.(?:tex|bib|sty|cls))\b")
LOG_RE = re.compile(r"(?m)^(?:!|l\.\d+|This is \w*TeX|\*\*|Latexmk|\(\./|Output written|Package \w+ (?:Warning|Error)|LaTeX (?:Warning|Error)|"
                    r"Runaway argument|[OU]\w+full \\|\S+\.\w+:\d+:|\$ |> |\w*latex(?:\.exe)? |pdffonts|set terminal|Font set to)")
FAMILY = {"tex": "src", "sty": "src", "cls": "src", "bib": "bib"}
_kpse: dict[str, bool] = {}


def in_texlive(name: str) -> bool:
    if name not in _kpse:
        _kpse[name] = subprocess.run(["kpsewhich", name], capture_output=True).returncode == 0
    return _kpse[name]


def classify(block: str) -> str | None:
    """bib / sty / cls / tex companion, None for logs, full documents and junk."""
    if mt.is_full_doc(block) and "{subfiles}" not in block:
        return None
    if re.search(r"(?m)^\s*@\w+\s*[{(]", block):
        return "bib"
    if "\\ProvidesPackage" in block:
        return "sty"
    if "\\ProvidesClass" in block:
        return "cls"
    if LOG_RE.search(block) or not re.search(r"\\[a-zA-Z]+", block):
        return None
    return "tex"


def blocks_with_hints(body: str) -> list[dict]:
    """Code blocks in order with a filename hint from the text just before the block or its first comment line."""
    out, prev = [], 0
    for m in mt.CODE_RE.finditer(body):
        code = mt.norm_code(m.group(1))
        pre = html.unescape(mt.TAG_RE.sub(" ", body[prev:m.start()]))[-300:]
        first = code.split("\n", 1)[0]
        hint = (NAME_RE.findall(first) if first.lstrip().startswith("%") else []) or NAME_RE.findall(pre)
        out.append({"code": code, "hint": hint[-1] if hint else None})
        prev = m.end()
    return out


def refs_of(doc: str) -> list[tuple[str, str]]:
    """(kind, relative path) of files the doc needs that are neither filecontents-provided nor in TeX Live."""
    provided = {Path(n.strip()).name for n in mt.FILECONTENTS_RE.findall(doc)}
    out = []
    for cmd, arg in REF_RE.findall(m1.strip_comments(doc)):
        for name in (n.strip() for n in arg.split(",") if n.strip()):
            kind = {"bibliography": "bib", "addbibresource": "bib", "usepackage": "sty", "documentclass": "cls"}.get(cmd, "tex")
            path = str(Path(name if Path(name).suffix else name + {"bib": ".bib", "sty": ".sty", "cls": ".cls", "tex": ".tex"}[kind]))
            if Path(path).name in provided or path.startswith(("/", "..")) or in_texlive(Path(path).name):
                continue
            if (kind, path) not in out:
                out.append((kind, path))
    return out


def assign(refs: list, blocks: list[dict]) -> dict[str, str] | None:
    """Map every needed file to one companion block: by hinted name first, then the unnamed blocks of the same family
    (tex/sty/cls vs bib) in order of appearance when their count matches."""
    comps = [(FAMILY[k], b) for k, b in ((classify(b["code"]), b) for b in blocks) if k]
    for _, b in comps:  # sty/cls blocks carry their own name
        m = re.search(r"\\Provides(Package|Class)\s*\{([^}]+)\}", b["code"])
        b["hint"] = m.group(2).strip() + (".sty" if m.group(1) == "Package" else ".cls") if m else b["hint"]
    files, used, names = {}, set(), {Path(p).name for _, p in refs}
    for kind, path in refs:
        for i, (fam, b) in enumerate(comps):
            if i not in used and fam == FAMILY[kind] and b["hint"] and Path(b["hint"]).name == Path(path).name:
                used.add(i); files[path] = b["code"]; break
    for fam in ("src", "bib"):
        rest = [p for k, p in refs if FAMILY[k] == fam and p not in files]
        free = [i for i, (f, b) in enumerate(comps) if i not in used and f == fam and (not b["hint"] or Path(b["hint"]).name not in names)]
        if rest and len(rest) != len(free):
            return None
        files.update((p, comps[i][1]["code"]) for p, i in zip(rest, free))
    return files


# --- parse ---

def parse(args) -> None:
    TM.mkdir(parents=True, exist_ok=True)
    stats, cands, t0 = Counter(), {}, time.monotonic()
    for a in mt.iter_rows():
        if a.get("PostTypeId") != "1":
            continue
        stats["questions"] += 1
        if stats["questions"] % 100000 == 0:
            log(f"  parsed {stats['questions']} questions, {len(cands)} candidates, {time.monotonic() - t0:.0f}s")
        body = a.get("Body", "")
        docs = [c for c in mt.code_blocks(body) if mt.is_full_doc(c) and "{subfiles}" not in c]
        if not docs:
            continue
        stats["q_mwe"] += 1
        text = a.get("Title", "") + "\n" + html.unescape(mt.TAG_RE.sub("\n", body))
        if not ERR_MENTION_RE.search(text):
            continue
        stats["q_error_mention"] += 1
        broken = max(docs, key=len)
        if broken.count("\n") > mt.MAX_LINES or len(broken) > mt.MAX_CHARS:
            continue
        blocks = [b for b in blocks_with_hints(body) if b["code"] != broken]
        if not any(classify(b["code"]) for b in blocks):
            continue
        stats["q_has_companion_block"] += 1
        errs = mt.error_messages(body)
        cands[int(a["Id"])] = {
            "question_id": int(a["Id"]), "title": html.unescape(a.get("Title", "")),
            "tags": re.findall(r"<([^<>]+)>", a.get("Tags", "")) or [t for t in a.get("Tags", "").strip("|").split("|") if t],
            "score": int(a.get("Score", 0)), "created": a.get("CreationDate", "")[:10], "error_message": errs[0] if errs else "",
            "broken": broken, "blocks": blocks, "accepted_answer_id": int(a["AcceptedAnswerId"]) if a.get("AcceptedAnswerId") else None,
            "accepted_blocks": [], "top_blocks": [], "top_answer_id": None, "top_score": 0, "accepted_score": 0,
        }
    log(f"  pass 1 done: {len(cands)} candidates; scanning answers")
    for a in mt.iter_rows():
        if a.get("PostTypeId") != "2" or (q := cands.get(int(a.get("ParentId", 0)))) is None:
            continue
        score, aid, blocks = int(a.get("Score", 0)), int(a["Id"]), mt.code_blocks(a.get("Body", ""))
        if aid == q["accepted_answer_id"]:
            q["accepted_blocks"], q["accepted_score"] = blocks, score
        elif score >= 2 and score > q["top_score"] and blocks:
            q["top_blocks"], q["top_answer_id"], q["top_score"] = blocks, aid, score
    out = [q for q in cands.values() if q["accepted_blocks"] or q["top_blocks"]]
    stats.update(has_accepted=sum(bool(q["accepted_blocks"]) for q in out), candidates=len(out))
    CANDS.write_text("".join(json.dumps(q) + "\n" for q in out))
    STATS.write_text(json.dumps({"parse": stats}))
    log(f"parse done: {dict(stats)}")


# --- mine ---

def keeps_companions(files: dict[str, str], fixed: dict[str, str]) -> bool:
    """The fix must stay multi-file: no new filecontents, every companion still referenced by name from another file."""
    if "\\begin{filecontents" in fixed["main.tex"] and "\\begin{filecontents" not in files["main.tex"]:
        return False
    return all(any(Path(f).stem in t for g, t in fixed.items() if g != f) for f in files if f != "main.tex")


FC_RE = re.compile(r"\\begin\{filecontents\*?\}\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}\n(.*?)\\end\{filecontents\*?\}\n?", re.S)


def split_filecontents(doc: str, files: dict[str, str]) -> list[tuple[str, str]]:
    """An answer that inlines companions as filecontents -> [(file, text)] for the companions it names plus the main doc without them."""
    out = []
    for m in FC_RE.finditer(doc):
        name = Path(m.group(1).strip()).name
        target = next((f for f in files if Path(f).name == name), None)
        if target:
            out.append((target, m.group(2)))
    return out + [("main.tex", re.sub(r"\\RequirePackage\{filecontents\}\n?", "", FC_RE.sub("", doc)))]


def fixed_versions(files: dict[str, str], blocks: list[str]) -> list[tuple[str, dict[str, str]]]:
    """(source, fixed files) plans from one answer: every block applied to the file it targets, then each alone."""
    per_block = []
    for b in blocks:
        kind, target = classify(b), None
        if mt.is_full_doc(b) and "{subfiles}" not in b:
            for target, text in split_filecontents(b, files):
                new = mt.merge_whitespace(files[target], text)
                if new != files[target]:
                    per_block.append(("answer_doc", target, new))
            continue
        elif kind == "bib" or kind in ("sty", "cls"):
            m = re.search(r"\\Provides(?:Package|Class)\s*\{([^}]+)\}", b)
            names = [f for f in files if f.endswith("." + kind) and (not m or Path(f).stem == m.group(1).strip())]
            target = names[0] if len(names) == 1 else None
            if target and kind == "bib" and len(re.findall(r"(?m)^@\w+", b)) < len(re.findall(r"(?m)^@\w+", files[target])):
                new = mt.splice_snippet(files[target], b)  # partial bib: splice the entries
            else:
                new = mt.merge_whitespace(files[target], b) if target else None
        elif kind == "tex":
            keys = {mt.wskey(l) for l in b.split("\n") if l.strip()}
            scored = sorted(((sum(mt.wskey(l) in keys for l in t.split("\n")), f == "main.tex", f) for f, t in files.items()
                             if f.endswith((".tex", ".sty", ".cls"))), reverse=True)
            target = scored[0][2] if scored and scored[0][0] else None
            new = mt.merge_whitespace(files[target], s) if target and (s := mt.splice_snippet(files[target], b)) else None
        else:
            new = None
        if target and new and new != files[target]:
            per_block.append(("snippet", target, new))
    plans, seen = [], set()
    if len(per_block) > 1 and len({t for _, t, _ in per_block}) == len(per_block):
        plans.append(("multi", {**files, **{t: n for _, t, n in per_block}}))
    plans += [(src, {**files, t: n}) for src, t, n in per_block]
    out = []
    for src, fx in plans:
        key = json.dumps(fx, sort_keys=True)
        if key not in seen and fx["main.tex"].count("\\documentclass") == 1 and keeps_companions(files, fx):
            seen.add(key); out.append((src, fx))
    return out[:4]


def check(src: Path, files: dict, broken: dict, edits: list, bib: bool) -> dict:
    """mine_github_multi.check_pair with the bib pipeline forced on for .bib companions (which TeX itself never opens)."""
    if not bib:
        return m1.check_pair(src, files, broken, edits, timeout=90)
    rc = common.compile_project(src, [], timeout=90, bib=True)
    if not rc["ok"]:
        return {"stage": "child_timeout" if rc["timed_out"] else "child_compile_fail"}
    rb = common.compile_project(src, [], overrides={f: broken[f] for f in broken if broken[f] != files[f]}, timeout=90, bib=True)
    if rb["timed_out"] or rb["ok"]:
        return {"stage": "parent_timeout" if rb["timed_out"] else "parent_compiles"}
    row = m1.make_row(src, files, broken, edits, rb, {"bib": True})
    if not row or not row["error_lines"]:
        return {"stage": "parent_no_located_error"}
    return {"stage": "kept", "row": row, "engine": rc["engine"], "ref": {"text": rc["pdf_text"], "pages": rc["pdf_pages"]}}


def process(c: dict) -> dict:
    qid, broken_main = c["question_id"], c["broken"]
    refs = refs_of(broken_main)
    if not refs:
        return {"stage": "no_external_ref"}
    comps = assign(refs, c["blocks"])
    if comps is None:
        return {"stage": "companion_unmatched"}
    if len(comps) > 6:
        return {"stage": "too_many_files"}
    broken = {"main.tex": broken_main, **comps}
    if any(p not in broken for t in comps.values() for _, p in refs_of(t)):
        return {"stage": "companion_needs_more"}
    plans, why = [], Counter()
    for kind, blocks, aid, ascore in (("accepted", c["accepted_blocks"], c["accepted_answer_id"], c["accepted_score"]),
                                      ("top", c["top_blocks"], c["top_answer_id"], c["top_score"])):
        for src, fixed in fixed_versions(broken, blocks):
            edits = []
            for f in sorted(fixed):
                if fixed[f] != broken[f]:
                    es, w = mt.derive_edits(fixed[f], broken[f])
                    if es is None:
                        why[w] += 1; edits = None; break
                    edits += [(f, e["search"], e["replace"], "real") for e in es]
            if edits and sum(len(s.split("\n")) + len(r.split("\n")) for _, s, r, _ in edits) <= 80 and len({e[0] for e in edits}) <= 4:
                plans.append((kind, aid, ascore, src, fixed, edits))
            elif edits:
                why["diff_too_big"] += 1
    if not plans:
        return {"stage": "fix_" + (why.most_common(1)[0][0] if why else "no_fix_candidate")}
    last = "fixed_compile_fail"
    for kind, aid, ascore, src, fixed, edits in plans:
        d = mt.materialize(qid, fixed["main.tex"], broken_main, base=TM)
        for f, t in fixed.items():
            (d / f).parent.mkdir(parents=True, exist_ok=True); (d / f).write_text(t)
            for m in mt.GRAPHICS_RE.finditer(t):
                ref = Path(m.group(1).strip()).name
                tgt = d / (ref if "." in ref else ref + ".pdf")
                if not ref.startswith("example-") and not tgt.exists():
                    tgt.write_bytes(mt.PNG_BYTES if tgt.suffix.lower() in (".png", ".jpg", ".jpeg") else mt.PDF_BYTES)
        files = common.project_files(d)
        res = check(d, files, {**files, **broken}, edits, bib=any(f.endswith(".bib") for f in comps))
        if res["stage"] == "kept":
            row = res["row"]
            row.update({
                "id": f"texse_multi_{qid}", "doc": f"texse_multi/{qid}", "src_dir": str(d), "mutation": "real", "seed": 0, "source": "texse",
                "license": mt.license_of(c["created"]), "url": f"https://tex.stackexchange.com/q/{qid}", "fix_answer": kind, "answer_id": aid, "answer_score": ascore,
                "fix_source": src, "companions": sorted(comps), "n_files": sum(1 for f in files if Path(f).suffix in m1.SRC_EXTS),
                "engine": res["engine"], "error_category": mt.normalize_error(row["error_lines"][0]["text"]), "changed_files": row["fix_files"],
                **{k: c[k] for k in ("question_id", "accepted_answer_id", "title", "tags", "score", "created", "error_message")},
            })
            return {"stage": "kept", "row": row, "ref": res["ref"]}
        last = res["stage"]
        shutil.rmtree(d.parent, ignore_errors=True)
        if last in ("child_compile_fail", "child_timeout"):
            continue
        break  # broken project itself is unusable: no plan will change that
    return {"stage": last}


def mine(args) -> None:
    cands = [json.loads(l) for l in CANDS.read_text().splitlines()]
    used = {json.loads(l)["question_id"] for n in ("texse_eval", "texse2_labeled", "texse_pool", "texse2_pool")
            for l in (common.DATA / f"{n}.jsonl").read_text().splitlines()}
    done = {d["question_id"] for d in map(json.loads, DONE.read_text().splitlines()) if d["stage"] != "worker_error"} if DONE.exists() else set()
    stats = Counter(json.loads(STATS.read_text()).get("mine", {})) if STATS.exists() else Counter()
    stats["used_elsewhere"] = stats["dedup"] = 0  # recounted below
    seen, todo = set(), []
    for c in cands:
        if c["question_id"] in used:
            stats["used_elsewhere"] += 1
        elif (h := mwe_hash(c["broken"])) in seen:
            stats["dedup"] += 1
        else:
            seen.add(h)
            if c["question_id"] not in done:
                todo.append(c)
    todo = todo[:args.limit] if args.limit else todo
    log(f"mine: {len(todo)} candidates ({len(done)} done before, {stats['used_elsewhere']} used elsewhere), {args.workers} workers")
    t0, n = time.monotonic(), 0
    with ThreadPoolExecutor(args.workers) as ex, DONE.open("a") as df, OUT.open("a") as of, REFS.open("a") as rf:
        futs = {ex.submit(process, c): c for c in todo}
        for fut in as_completed(futs):
            c, n = futs[fut], n + 1
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = {"stage": "worker_error"}; log(f"  q{c['question_id']} error: {e!r}"[:200])
            stats[res["stage"]] += 1
            df.write(json.dumps({"question_id": c["question_id"], "stage": res["stage"]}) + "\n"); df.flush()
            if res["stage"] == "kept":
                of.write(json.dumps(res["row"]) + "\n"); of.flush()
                rf.write(json.dumps({"doc": res["row"]["doc"], "ref": res["ref"]}) + "\n"); rf.flush()
                log(f"  KEPT q{c['question_id']} [{res['row']['error_category']}] fix={res['row']['fix_files']} reported={res['row']['reported_file']} files={res['row']['companions']}")
            if n % 25 == 0 or n == len(todo):
                rate = n / (time.monotonic() - t0)
                log(f"{n}/{len(todo)} kept={stats['kept']} eta {(len(todo) - n) / max(rate, 1e-6) / 60:.0f}m  {dict(stats.most_common(8))}")
                STATS.write_text(json.dumps({**json.loads(STATS.read_text()), "mine": dict(stats)}))
    log(f"mine done: {dict(stats)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parse", "mine"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    {"parse": parse, "mine": mine}[args.cmd](args)


if __name__ == "__main__":
    main()
