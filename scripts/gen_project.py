"""v3: project-level perturbations (multi-file arXiv projects). The error is often reported in
a different file than the one that needs the fix. Rows carry `file_edits` (break edits on named
files); targets are <edit file="..."> blocks. Single-shot prompt = tree + reported file window;
`single_shot_visible` says whether all fix sites are in the prompt (else it is agent-lane only).

Usage: gen_project.py --docs-glob 'arxiv_multi/*' --per-doc 6 --out ../data/project_eval.jsonl
"""
import argparse
import json
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import common
import perturb
from gen_data import V2_FULL_LINES, V2_MAX_CHARS

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GFX_RE = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}")
BIBSTYLE_RE = re.compile(r"\\bibliographystyle\{([^}]+)\}")
BIB_ENTRY_RE = re.compile(r"^@\w+\s*\{[^,]+,\n(?:.*\n)*?\}\s*$", re.M)
DEF_RE = re.compile(r"^[ \t]*\\(?:newcommand|renewcommand|providecommand|def)\s*\\([a-zA-Z]+)[^\n]*\{([^\n]*)\}[ \t]*$", re.M)


def _swap(word: str, rng) -> str:
    i = rng.randrange(len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _unique(text: str, s: str) -> bool:
    return text.count(s) == 1


def children(files: dict) -> list[str]:
    """Child .tex files really pulled in by main.tex (comment-stripped, so a commented-out
    \\input does not count — perturbing an unused file compiles fine and wastes a task)."""
    live = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in files["main.tex"].splitlines())
    out = []
    for m in INPUT_RE.finditer(live):
        rel = m.group(1).strip()
        rel = rel if rel.endswith(".tex") else rel + ".tex"
        if rel in files:
            out.append(rel)
    return out


# each mutation: (files, rng, meta) -> list[(file, search, replace)] or None

def child_error(files, rng, meta):
    kids = children(files)
    if not kids:
        return None
    kid = rng.choice(kids)
    fn = rng.choice(perturb.MUTATIONS)
    e = fn(files[kid], random.Random(rng.randrange(1 << 30)))
    return [(kid, e["search"], e["replace"], "child_" + e["type"])] if e else None


def sty_macro(files, rng, meta):
    """Typo inside a macro body defined in a project .sty/.cls; symptom appears where it is used."""
    # project-local macro files first (a user's own .sty), vendor .cls last
    for sty in sorted((f for f in files if f.endswith((".sty", ".cls"))),
                      key=lambda f: (f.endswith(".cls"), "macro" not in f.lower())):
        for m in list(DEF_RE.finditer(files[sty]))[:50]:
            name, body = m.group(1), m.group(2)
            uses = sum(len(re.findall(r"\\" + name + r"\b", t)) for f, t in files.items() if f != sty)
            inner = re.search(r"\\([a-zA-Z]{4,})", body)
            if uses < 1 or not inner or not _unique(files[sty], m.group(0)):
                continue
            bad_line = m.group(0).replace("\\" + inner.group(1), "\\" + _swap(inner.group(1), rng), 1)
            if bad_line != m.group(0) and _unique(files[sty].replace(m.group(0), bad_line, 1), bad_line):
                return [(sty, m.group(0), bad_line, "sty_macro")]
    return None


def input_path(files, rng, meta):
    ms = list(INPUT_RE.finditer(files["main.tex"]))
    rng.shuffle(ms)
    for m in ms:
        path = m.group(1)
        stem = Path(path).stem
        if len(stem) < 4:
            continue
        bad = m.group(0).replace(stem, _swap(stem, rng), 1)
        if _unique(files["main.tex"], m.group(0)) and bad != m.group(0):
            return [("main.tex", m.group(0), bad, "input_path")]
    return None


def graphics_path(files, rng, meta):
    for f in rng.sample(list(files), len(files)):
        if not f.endswith(".tex"):
            continue
        for m in list(GFX_RE.finditer(files[f]))[:20]:
            ref = m.group(1).strip()
            stem = Path(ref).stem
            if len(stem) < 4 or not _unique(files[f], m.group(0)):
                continue
            bad = m.group(0).replace(stem, _swap(stem, rng), 1)
            if bad != m.group(0):
                return [(f, m.group(0), bad, "graphics_path")]
    return None


def bib_style(files, rng, meta):
    if not meta.get("bib_ok"):
        return None
    m = BIBSTYLE_RE.search(files["main.tex"])
    if not m or not _unique(files["main.tex"], m.group(0)):
        return None
    bad = m.group(0).replace(m.group(1), _swap(m.group(1), rng), 1)
    return [("main.tex", m.group(0), bad, "bib_style")] if bad != m.group(0) else None


def bib_entry(files, rng, meta):
    """Break a cited .bib entry (drop its closing brace) -> bibtex fails."""
    if not meta.get("bib_ok"):
        return None
    cited = set(k for t in files.values() for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", t)
                for k in m.group(1).split(","))
    for bib in [f for f in files if f.endswith(".bib")]:
        entries = [m for m in BIB_ENTRY_RE.finditer(files[bib])
                   if re.match(r"@\w+\s*\{([^,]+),", m.group(0)).group(1).strip() in cited]
        rng.shuffle(entries)
        for m in entries[:10]:
            ent = m.group(0)
            bad = ent.rstrip()[:-1].rstrip() + "\n"  # remove final }
            if _unique(files[bib], ent):
                return [(bib, ent, bad, "bib_entry")]
    return None


def preamble_drop(files, rng, meta):
    """Drop a \\usepackage in main.tex whose commands are used only in a child file."""
    kids = children(files)
    if not kids:
        return None
    pk = list(perturb.USEPACKAGE_RE.finditer(files["main.tex"]))
    rng.shuffle(pk)
    for m in pk[:8]:
        line = m.group(0)
        got = perturb._unique_window(files["main.tex"], m.start(), perturb._line_span(files["main.tex"], m.end())[1])
        if not got:
            continue
        search, ws, we = got
        replace = search[: m.start() - ws] + search[m.end() - ws:]
        if replace.strip() and _unique(files["main.tex"].replace(search, replace, 1), replace):
            return [("main.tex", search, replace, "preamble_drop")]
    return None


PROJECT_MUTATIONS = [child_error, sty_macro, input_path, graphics_path, bib_style, bib_entry, preamble_drop]


def apply(files: dict, edits) -> dict:
    files = dict(files)
    for f, s, r, _ in edits:
        files[f] = files[f].replace(s, r, 1)
    return files


def make_break(files: dict, k: int, seed: int, meta: dict):
    rng = random.Random(f"{meta.get('slug', '')}:{seed}")  # per-doc stream, else every doc draws the same classes
    fns = rng.sample(PROJECT_MUTATIONS, k)
    cur, edits = files, []
    for fn in fns:
        e = fn(cur, rng, meta)
        if not e:
            continue
        nxt = apply(cur, e)
        if all(nxt[f].count(r) == 1 for f, s, r, _ in e):
            cur, edits = nxt, edits + e
    if not edits:
        return None
    back = cur
    for f, s, r, _ in reversed(edits):
        if back[f].count(r) != 1:
            return None
        back = {**back, f: back[f].replace(r, s, 1)}
    return (cur, edits) if back == files else None


def gen_one(doc: dict, k: int, seed: int) -> dict | None:
    try:
        return _gen_one(doc, k, seed)
    except (FileNotFoundError, OSError):  # project pruned by a concurrent ingester
        return None


def _gen_one(doc: dict, k: int, seed: int) -> dict | None:
    src = Path(doc["src_dir"])
    files = common.project_files(src)
    if "main.tex" not in files:
        return None
    got = make_break(files, k, seed, doc)
    if not got:
        return None
    broken, edits = got
    overrides = {f: broken[f] for f in broken if broken[f] != files[f]}
    r = common.compile_project(src, [], overrides=overrides, timeout=120, bib=doc.get("bib_ok", False))
    if r["ok"]:
        return None
    types = "+".join(t for _, _, _, t in edits)
    return make_row(src, files, broken, edits, r, {
        "id": f"{doc['slug'].replace('/', '_')}__{types}__{seed}", "doc": doc["slug"], "src_dir": doc["src_dir"],
        "mutation": types, "seed": seed, "bib": bool(doc.get("bib_ok"))})


def make_row(src: Path, files: dict, broken: dict, edits: list, r: dict, base: dict) -> dict | None:
    """Project-lane row from a compiled break: prompt = tree + reported-file window, target = reverse edits.
    edits = [(file, search=good text, replace=broken text, type)]; r = compile result of `broken`."""
    err_lines = common.error_lines_from_items(r["items"], "main.tex", set(files) | {"main.tex"})
    if not err_lines and not r["error_items"]:
        return None
    reported = (err_lines[0]["file"] if err_lines and err_lines[0]["file"] else "main.tex")
    tree = common.tree_block(src)
    prompt = common.build_fix_prompt(reported, err_lines, r["log"], broken.get(reported, ""),
                                     full_lines=V2_FULL_LINES, max_chars=V2_MAX_CHARS, tree=tree)
    fix = [(f, r_, s) for f, s, r_, _ in reversed(edits)]
    target = common.format_file_edits(fix)
    fixed, err = common.apply_file_edits(broken, common.parse_file_edits(target))
    if err or fixed != files:
        return None
    visible = all(any(l.strip()[:60] in prompt for l in (r_.splitlines() or [s]) if len(l.strip()) >= 8)
                  or not any(len(l.strip()) >= 8 for l in r_.splitlines()) for f, s, r_, _ in edits)
    return {
        **base, "bundles": [], "lane": "project",
        "file_edits": [{"file": f, "search": s, "replace": r_, "type": t} for f, s, r_, t in edits],
        "reported_file": reported, "fix_files": sorted({f for f, *_ in edits}),
        "single_shot_visible": visible, "error_lines": err_lines,
        "log_tail": r["log"][-common.MAX_LOG_CHARS:], "prompt": prompt, "target": target,
    }


def baseline(doc: dict) -> dict:
    src = Path(doc["src_dir"])
    try:
        r0 = common.compile_project(src, [], timeout=120)
    except (FileNotFoundError, OSError):
        return {**doc, "ok": False}
    if not r0["ok"]:
        return {**doc, "ok": False}
    r1 = common.compile_project(src, [], timeout=150, bib=True)
    return {**doc, "ok": True, "bib_ok": r1["ok"], "seconds": r0["seconds"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-glob", default="arxiv_multi/*")
    ap.add_argument("--per-doc", type=int, default=6)
    ap.add_argument("--max-k", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(common.DATA / "project_eval.jsonl"))
    ap.add_argument("--cap-per-type", type=int, default=60)
    ap.add_argument("--exclude-docs-file", default="", help="JSON list of project slugs to skip")
    args = ap.parse_args()

    docs = [{"slug": f"{p.parent.name}/{p.name}", "src_dir": str(p / "files")}
            for p in sorted((common.PROJ / "corpus").glob(args.docs_glob)) if (p / "files" / "main.tex").exists()]
    if args.exclude_docs_file:
        banned = set(json.loads(Path(args.exclude_docs_file).read_text()))
        docs = [d for d in docs if d["slug"] not in banned]
    print(f"{len(docs)} projects; baseline (plain + bib)...")
    good = []
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(baseline, d) for d in docs]):
            r = f.result()
            if r["ok"]:
                good.append(r)
    print(f"baseline ok: {len(good)}/{len(docs)}; bib pipeline ok: {sum(d['bib_ok'] for d in good)}")

    rng = random.Random(3)
    tasks = [(d, rng.randint(1, args.max_k), s) for d in good for s in range(args.per_doc)]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    kept, by_type, t0 = 0, {}, time.monotonic()
    with out.open("w") as fh, ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(gen_one, d, k, s) for d, k, s in tasks]
        for i, f in enumerate(as_completed(futs)):
            row = f.result()
            if row and by_type.get(row["mutation"].split("+")[0], 0) >= args.cap_per_type:
                row = None  # balance: this class already has enough
            if row:
                fh.write(json.dumps(row) + "\n"); fh.flush(); kept += 1
                for t in row["mutation"].split("+"):
                    by_type[t] = by_type.get(t, 0) + 1
            if (i + 1) % 20 == 0:
                rate = (i + 1) / (time.monotonic() - t0)
                print(f"  {i + 1}/{len(tasks)} done, {kept} kept, eta {(len(tasks) - i - 1) / rate / 60:.0f}m")
    print(f"\nwrote {kept} examples to {out}")
    for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {t:22s} {n}")


if __name__ == "__main__":
    main()
