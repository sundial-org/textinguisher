"""Sample modern arXiv papers from scholarweave/arxiv-latex (HF datasets-server rows API),
materialize each as corpus/arxiv/<id>/files/ (root renamed main.tex, placeholder figures),
keep the ones that baseline-compile clean. No HF libraries needed.

Usage: gen_arxiv.py --target 150 [--seed 11] [--workers 6]
"""
import argparse
import json
import random
import re
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import common

ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=scholarweave%2Farxiv-latex"
            "&config=default&split=train&offset={offset}&length={length}")
N_ROWS = 3_100_000  # approx dataset size; we sample high offsets for modern LaTeX
MIN_OFFSET = 2_200_000
FILE_HEADER_RE = re.compile(r"^={10,}\nFILE: (.+?)\n={10,}\n", re.M)

# 1x1 transparent PNG + minimal one-page PDF, used as placeholder figures.
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478"
    "9c626001000000ffff03000006000557bfabd40000000049454e44ae426082")
PDF_BYTES = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
             b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
             b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 72 72]>>endobj\n"
             b"trailer<</Root 1 0 R>>\n%%EOF\n")


def fetch_rows(offset: int, length: int = 25) -> list[dict]:
    req = urllib.request.Request(ROWS_URL.format(offset=offset, length=length),
                                 headers={"User-Agent": "latex-fix-datagen"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return [r["row"] for r in json.load(resp)["rows"]]


def split_files(latex: str) -> dict[str, str]:
    parts = FILE_HEADER_RE.split(latex)
    if len(parts) < 3:
        return {}
    files = {}
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip()
        if not name or name.startswith("/") or ".." in name:
            continue
        files[name] = parts[i + 1]
    return files


MULTI_ONLY = False  # set by --multi-only
GRAPHICS_RE = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}")


def materialize(row: dict, out_root: Path, max_chars: int = 300_000) -> Path | None:
    """Write the paper's files (+placeholder figures) into out_root/<id>/files. Returns dir or None."""
    latex = row.get("latex") or ""
    if not (10_000 < len(latex) < max_chars):
        return None
    files = split_files(latex)
    roots = [n for n, t in files.items()
             if n.lower().endswith(".tex") and "\\documentclass" in t and "\\begin{document}" in t]
    if len(roots) != 1 or any("\\psfig" in t or "\\special{" in t for t in files.values()):
        return None
    root = roots[0]
    if MULTI_ONLY:
        n_tex = sum(1 for n in files if n.lower().endswith(".tex"))
        if n_tex < 2 or not re.search(r"\\(?:input|include)\{", files[root]):
            return None
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", row["id"])
    d = out_root / slug / "files"
    if d.exists():
        shutil.rmtree(d.parent)
    d.mkdir(parents=True)
    for name, text in files.items():
        rel = "main.tex" if name == root else name.lstrip("./")  # keep subdirs: \input{sec/x} must resolve
        try:
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(text)
        except OSError:
            shutil.rmtree(d.parent)
            return None
    for m in GRAPHICS_RE.finditer("\n".join(files[n] for n in files if n.endswith(".tex"))):
        ref = m.group(1).strip().lstrip("./")
        if ".." in ref or ref.startswith("/"):
            continue
        target = d / (ref if Path(ref).suffix else ref + ".pdf")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            from gen_skew import placeholder  # valid PNG/JPEG/EPS bytes (PNG_BYTES is rejected by pdftex's libpng)
            target.write_bytes(placeholder(target))
    (d.parent / "meta.json").write_text(json.dumps(
        {"id": row["id"], "license": row.get("license"), "categories": row.get("categories")}))
    return d


def check_one(src_dir: str) -> tuple[str, bool, float]:
    r = common.compile_project(Path(src_dir), [], timeout=60)
    return src_dir, r["ok"], r["seconds"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default="arxiv", help="subdir of corpus/ to write into")
    ap.add_argument("--max-chars", type=int, default=300_000)
    ap.add_argument("--multi-only", action="store_true", help="keep only multi-file projects with \\input refs")
    args = ap.parse_args()
    global MULTI_ONLY
    MULTI_ONLY = args.multi_only

    out_root = common.PROJ / "corpus" / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    kept, tried = 0, 0

    while kept < args.target and tried < args.target * 30:
        offset = rng.randrange(MIN_OFFSET, N_ROWS - 30)
        try:
            rows = fetch_rows(offset)
        except Exception as e:  # noqa: BLE001
            print(f"fetch offset={offset} failed: {str(e)[:80]}")
            continue
        dirs = []
        for row in rows:
            tried += 1
            d = materialize(row, out_root, args.max_chars)
            if d:
                dirs.append(str(d))
        with ProcessPoolExecutor(args.workers) as ex:
            for f in as_completed([ex.submit(check_one, d) for d in dirs]):
                src_dir, ok, secs = f.result()
                slug = Path(src_dir).parent.name
                if ok and kept < args.target:
                    kept += 1
                    print(f"  keep {slug:24s} {secs:5.1f}s   [{kept}/{args.target}]")
                else:
                    shutil.rmtree(Path(src_dir).parent, ignore_errors=True)
        print(f"offset {offset}: kept so far {kept} (tried {tried})")
    print(f"done: {kept} compilable arxiv papers in {out_root}")


if __name__ == "__main__":
    main()
