"""Synthetic multi-file rows whose CAUSE is in a non-root file (source 2 of data/multi_report.md; labeled synthetic).

Seeds = compiling multi-file arXiv projects (corpus/arxiv_multi). Mutations are gen_project's non-root classes plus
new ones weighted after the real TeX.SE error histogram (undefined control sequence, missing number, missing $,
extra }, missing \\begin, pgfkeys unknown key, illegal parameter number). Rows = project-lane schema + provenance.

  gen_multi_synth.py --per-doc 12 --workers 8 --out ../data/multi_synth.jsonl   (resumable, incremental)
"""
import argparse
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import common
import gen_project
import perturb
from mine_texse import normalize_error

NUMBER_RE = re.compile(r"(\\(?:hspace|vspace|rule|setlength\{\\[a-zA-Z]+\}|addtolength\{\\[a-zA-Z]+\}|hskip|vskip|kern)\*?\{?)(-?\d*\.?\d+)(\s*(?:cm|em|ex|pt|mm|in|bp)\b)")
TIKZ_KEY_RE = re.compile(r"\b(line width|node distance|minimum width|minimum height|inner sep|outer sep|text width|"
                         r"rounded corners|column sep|row sep|xshift|yshift|opacity|anchor|fill|draw|scale|font|"
                         r"xlabel|ylabel|legend pos|ymode|xmode|width|height|mark|color)\s*=")
ARGDEF_RE = re.compile(r"^[ \t]*\\(?:newcommand|renewcommand|providecommand)\*?\{?\\([a-zA-Z]+)\}?\[(\d)\][^\n]*\{[^\n]*#1[^\n]*\}[ \t]*$", re.M)
COMMENT_RE = re.compile(r"^[ \t]*%+[ \t]*([A-Za-z][A-Za-z ,.'-]{20,})[ \t]*$", re.M)


def nonroot(files: dict) -> list[str]:
    return gen_project.children(files) + [f for f in files if f.endswith((".sty", ".cls"))]


def _child(fn):
    """Lift a single-file perturb mutation onto a random \\input child of main.tex."""
    def m(files, rng, meta):
        kids = gen_project.children(files)
        if not kids:
            return None
        kid = rng.choice(kids)
        e = fn(files[kid], random.Random(rng.randrange(1 << 30)))
        return [(kid, e["search"], e["replace"], "child_" + e["type"])] if e else None
    m.__name__ = "child_" + fn.__name__
    return m


def _line_mutation(files, rng, regex, mutate, mtype, targets):
    """Regex-driven mutation on a unique line window of a non-root file (same anchoring as perturb)."""
    for f in rng.sample(targets, len(targets)):
        ms = list(regex.finditer(files[f]))
        rng.shuffle(ms)
        for m in ms[:12]:
            e = perturb._window_edit(files[f], m.start(), lambda w, off, m=m: mutate(w, off, m, rng), mtype, rng)
            if e:
                return [(f, e["search"], e["replace"], mtype)]
    return None


def missing_number(files, rng, meta):  # \hspace{2cm} -> \hspace{cm}: "Missing number, treated as zero"
    def mut(w, off, m, rng):
        return w.replace(m.group(0), m.group(1) + m.group(3), 1)
    return _line_mutation(files, rng, NUMBER_RE, mut, "missing_number", nonroot(files))


def pgfkeys_key(files, rng, meta):  # line width= -> line widht=: "Package pgfkeys Error: I do not know the key"
    tik = [f for f in nonroot(files) if re.search(r"\\begin\{(?:tikzpicture|axis)\}|\\tikz\b|\\pgfplotsset", files[f])]
    def mut(w, off, m, rng):
        key = m.group(1)
        return w.replace(m.group(0), gen_project._swap(key, rng) + m.group(0)[len(key):], 1)
    return _line_mutation(files, rng, TIKZ_KEY_RE, mut, "pgfkeys_key", tik) if tik else None


def illegal_param(files, rng, meta):  # \newcommand{\x}[1]{..#1..} -> drops [1]: "Illegal parameter number"
    def mut(w, off, m, rng):
        return w.replace(f"[{m.group(2)}]", "", 1)
    return _line_mutation(files, rng, ARGDEF_RE, mut, "illegal_param", nonroot(files))


def sty_stray_text(files, rng, meta):  # comment loses its %: text in a .sty -> "Missing \begin{document}"
    stys = [f for f in files if f.endswith((".sty", ".cls"))]
    def mut(w, off, m, rng):
        return w.replace(m.group(0), m.group(1), 1)
    return _line_mutation(files, rng, COMMENT_RE, mut, "sty_stray_text", stys) if stys else None


def graphics_nonroot(files, rng, meta):
    e = gen_project.graphics_path(files, rng, meta)
    return e if e and e[0][0] != "main.tex" else None


# weights ~ real TeX.SE histogram (undefined control sequence 4x, missing number 2x, ...)
WEIGHTED = ([_child(perturb.typo_macro)] * 3 + [gen_project.sty_macro] * 3 + [missing_number] * 2
            + [_child(perturb.drop_dollar)] * 2 + [_child(perturb.extra_brace)] * 2 + [_child(perturb.drop_brace)]
            + [_child(perturb.env_mismatch)] + [_child(perturb.typo_begin_env)] + [sty_stray_text] * 2
            + [pgfkeys_key] * 2 + [illegal_param] * 2 + [_child(perturb.extra_amp)] + [_child(perturb.drop_newcommand)]
            + [gen_project.bib_entry] + [graphics_nonroot])


def make_break(files: dict, k: int, seed: int, meta: dict):
    """gen_project.make_break, but inapplicable draws fall through to the next class (repeats = weights)."""
    rng = random.Random(f"{meta.get('slug', '')}:{seed}")
    cur, edits = files, []
    for fn in rng.sample(WEIGHTED, len(WEIGHTED)):
        if len(edits) >= k:
            break
        e = fn(cur, rng, meta)
        if e:
            nxt = gen_project.apply(cur, e)
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


gen_project.make_break = make_break


def gen_one(doc: dict, seed: int) -> dict | None:
    k = 2 if seed % 5 == 4 else 1
    row = gen_project.gen_one(doc, k, seed)
    if not row:
        return None
    row.update({"source": "multi_synth", "synthetic": True, "id": "synth_" + row["id"],
                "error_category": normalize_error(row["error_lines"][0]["text"]) if row["error_lines"] else "unlocated",
                "n_files": sum(1 for f in common.project_files(doc["src_dir"]) if f.endswith((".tex", ".sty", ".cls", ".bib"))),
                "cause_elsewhere": row["reported_file"] not in row["fix_files"]})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-glob", default="arxiv_multi/*")
    ap.add_argument("--per-doc", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(common.DATA / "multi_synth.jsonl"))
    args = ap.parse_args()
    out = Path(args.out)
    done = {(r["doc"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    docs = [{"slug": f"{p.parent.name}/{p.name}", "src_dir": str(p / "files")}
            for p in sorted((common.PROJ / "corpus").glob(args.docs_glob)) if (p / "files" / "main.tex").exists()]
    base_path = out.with_suffix(".baseline.json")
    base = json.loads(base_path.read_text()) if base_path.exists() else {}
    todo = [d for d in docs if d["slug"] not in base]
    print(f"{len(docs)} projects; baseline for {len(todo)}...", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(gen_project.baseline, d) for d in todo]):
            r = f.result(); base[r["slug"]] = r
    base_path.write_text(json.dumps(base))
    good = [b for b in base.values() if b["ok"]]
    tasks = [(d, s) for d in good for s in range(args.per_doc) if (d["slug"], s) not in done]
    print(f"baseline ok {len(good)}/{len(docs)}; {len(tasks)} tasks ({len(done)} done before)", flush=True)
    kept, by_type, t0 = len(done), Counter(), time.monotonic()
    with out.open("a") as fh, ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(gen_one, d, s) for d, s in tasks]
        for i, f in enumerate(as_completed(futs)):
            row = f.result()
            if row:
                fh.write(json.dumps(row) + "\n"); fh.flush(); kept += 1
                by_type[row["error_category"]] += 1
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.monotonic() - t0)
                print(f"  {i + 1}/{len(tasks)} done, {kept} kept, eta {(len(tasks) - i - 1) / rate / 60:.0f}m", flush=True)
    print(f"\n{kept} rows in {out}")
    for t, n in by_type.most_common(15):
        print(f"  {n:4d} {t}")


if __name__ == "__main__":
    main()
