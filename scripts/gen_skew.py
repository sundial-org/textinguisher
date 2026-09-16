"""Organic hard eval: older arXiv sources (2008-2018) that fail under TeX Live 2025 with NO
planted error (package/class drift, option clashes, deprecated packages, encoding).
No ground truth -> compile-only scoring; rows have no break_edits.

  gen_skew.py --target 200 --workers 3      # sample + compile -> data/skew_eval.jsonl
  gen_skew.py --report                      # data/skew_report.md (+ probe numbers if present)
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import common
from gen_arxiv import PDF_BYTES, PNG_BYTES, fetch_rows, materialize

OUT_ROOT = common.PROJ / "corpus" / "arxiv_skew"
DATA = common.DATA / "skew_eval.jsonl"
STATS = common.DATA / "skew_stats.json"
REPORT = common.DATA / "skew_report.md"
LOG = common.PROJ / "out" / "gen_skew.log"
MIN_OFFSET, MAX_OFFSET = 500_000, 1_450_000  # verified: 500k=0810.xxxx, 1.45M~1812.xxxx
CS_RE = re.compile(r"(\\[A-Za-z@]+|\\.)\s*$")
FILE_NOT_FOUND_RE = re.compile(r"File `([^']+)' not found|Cannot find file `([^']+)'|Could not open file (\S+),")
PKG_EXT = (".sty", ".cls", ".bst", ".def", ".fd", ".cfg", ".clo")
# materialize() writes PDF bytes into .eps placeholders; TeX Live's restricted shell-escape
# converts real EPS fine, so give it a valid one to avoid a harness-only failure.
EPS_BYTES = (b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 72 72\n%%EndComments\n"
             b"newpath 0 0 moveto 72 0 lineto 72 72 lineto closepath fill\n%%EOF\n")
# gen_arxiv.PNG_BYTES is rejected by pdftex's libpng ("internal error"); valid 1x1 from gs.
GOOD_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de000000017352474201d9c92c7f"
    "000000097048597300000b1300000b1301009a9c180000000c49444154089963f8ffff3f0005fe02fe58f26b0e"
    "0000000049454e44ae426082")
_JPG: list[bytes] = []


def placeholder(path: Path) -> bytes:
    ext = path.suffix.lower()
    if ext in (".eps", ".ps"):
        return EPS_BYTES
    if ext in (".jpg", ".jpeg"):
        if not _JPG:  # pdftex picks the reader by extension, so .jpg needs real JPEG bytes
            out = path.parent / ".ph.jpg"
            subprocess.run(["gs", "-q", "-sDEVICE=jpeg", "-o", str(out), "-g1x1", "-c", "showpage"],
                           capture_output=True, timeout=60)
            _JPG.append(out.read_bytes() if out.exists() else GOOD_PNG)
            out.unlink(missing_ok=True)
        return _JPG[0]
    return GOOD_PNG


def fix_placeholders(d: Path) -> bool:
    """Replace materialize()'s broken image placeholders with valid ones. Returns True if any."""
    hit = False
    for f in d.rglob("*"):
        if f.is_file() and f.suffix.lower() in (".eps", ".ps", ".png", ".jpg", ".jpeg") and f.read_bytes() in (PDF_BYTES, PNG_BYTES):
            f.write_bytes(placeholder(f)); hit = True
    return hit


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def year_of(arxiv_id: str) -> int | None:
    m = re.match(r"^(\d{2})(\d{2})\.\d+", arxiv_id) or re.search(r"/(\d{2})\d{5}", arxiv_id)
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy > 90 else 2000 + yy


def signature(err: dict) -> str:
    """Normalized first-error message; undefined control sequences keep the macro name."""
    msg = err["message"]
    if not FILE_NOT_FOUND_RE.search(msg):  # missing files keep their name: one bin per package
        msg = re.sub(r"`[^']*'", "`<x>'", msg)
    msg = msg.strip().lower()
    if "u+fffd" not in msg:  # U+FFFD = bytes lost upstream, its own bin; other chars = one bin
        msg = re.sub(r"unicode character .+? \(u\+[0-9a-f]+\)", "unicode character <c> (u+<n>)", msg)
    msg = re.sub(r"\d+", "<n>", msg)
    if "undefined control sequence" in msg:
        src = next((l for l in err["rawExcerpt"].split("\n") if re.match(r"^l\.\d+", l)), "")
        m = CS_RE.search(src)
        if m:
            msg += " " + m.group(1)
    return msg[:100]


def missing_file(err: dict) -> str | None:
    m = FILE_NOT_FOUND_RE.search(err["rawExcerpt"])
    return next(g for g in m.groups() if g) if m else None


def compile_one(src_dir: str) -> dict:
    r = common.compile_project(Path(src_dir), [], timeout=90)
    r.pop("pdf_text", None)
    return {"src_dir": src_dir, **r}


def make_row(src_dir: Path, r: dict) -> dict:
    project_files = {p.name for p in src_dir.iterdir()}
    error_lines = common.error_lines_from_items(r["items"], "main.tex", project_files)
    main = (src_dir / "main.tex").read_text(errors="replace")
    first = r["error_items"][0]
    meta = json.loads((src_dir.parent / "meta.json").read_text())
    slug = src_dir.parent.name
    return {
        "id": f"arxiv_skew_{slug}", "doc": f"arxiv_skew/{slug}", "src_dir": str(src_dir),
        "bundles": [], "mutation": "organic", "engine": r["engine"],
        "compile_seconds": r["seconds"], "error_lines": error_lines,
        "log_tail": r["log"][-common.MAX_LOG_CHARS:],
        "prompt": common.build_fix_prompt("main.tex", error_lines, r["log"], main,
                                          full_lines=500, max_chars=16000),
        "signature": signature(first), "first_error": first["message"][:200],
        "missing_file": missing_file(first), "n_errors": len(r["error_items"]),
        "arxiv_id": meta["id"], "year": year_of(meta["id"]),
    }


def generate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    funnel = Counter()
    sig_count: Counter = Counter()
    kept_ids: set[str] = set()
    if args.resume and DATA.exists():
        for line in DATA.read_text().splitlines():
            row = json.loads(line)
            sig_count[row["signature"]] += 1
            kept_ids.add(row["doc"].split("/", 1)[1])
        if STATS.exists():
            funnel.update(json.load(STATS.open()).get("funnel", {}))
        log(f"resume: {len(kept_ids)} rows, {len(sig_count)} signatures")
    else:
        DATA.write_text("")
        for p in OUT_ROOT.iterdir():
            shutil.rmtree(p, ignore_errors=True)
    kept = len(kept_ids)
    n_missing = sum(FILE_NOT_FOUND_RE.search(s) is not None for s in sig_count.elements())
    t0 = time.monotonic()
    with ProcessPoolExecutor(args.workers) as ex:
        while kept < args.target and funnel["sampled"] < args.max_sampled:
            offset = rng.randrange(MIN_OFFSET, MAX_OFFSET)
            try:
                rows = fetch_rows(offset)
            except Exception as e:  # noqa: BLE001
                log(f"fetch offset={offset} failed: {str(e)[:80]}")
                time.sleep(5)
                continue
            dirs = []
            for row in rows:
                funnel["sampled"] += 1
                slug = re.sub(r"[^A-Za-z0-9_.-]", "_", row["id"])
                if slug in kept_ids:
                    continue
                d = materialize(row, OUT_ROOT, args.max_chars)
                if d:
                    funnel["materialized"] += 1
                    fix_placeholders(d)
                    dirs.append(str(d))
            for f in as_completed([ex.submit(compile_one, d) for d in dirs]):
                r = f.result()
                src = Path(r["src_dir"])
                slug = src.parent.name
                verdict = "ok"
                if r["ok"]:
                    funnel["compiled_ok"] += 1
                elif r["timed_out"]:
                    funnel["timed_out"] += 1; verdict = "timeout"
                elif not r["error_items"]:
                    funnel["failed_no_error"] += 1; verdict = "fail-noerr"
                else:
                    funnel["compiled_failed"] += 1
                    row = make_row(src, r)
                    mf = row["missing_file"]
                    if not row["error_lines"]:
                        funnel["failed_unlocated"] += 1; verdict = "fail-unlocated"
                    elif mf and not mf.lower().endswith(PKG_EXT):
                        funnel["excluded_missing_input"] += 1; verdict = f"excluded-missing {mf}"
                    elif sig_count[row["signature"]] >= args.max_per_sig or (
                            mf and n_missing >= args.max_missing_frac * args.target):
                        funnel["capped"] += 1; verdict = f"capped [{row['signature'][:40]}]"
                    else:
                        sig_count[row["signature"]] += 1
                        n_missing += bool(mf)
                        kept += 1
                        kept_ids.add(slug)
                        funnel["kept"] += 1
                        with DATA.open("a") as f_out:
                            f_out.write(json.dumps(row) + "\n")
                        verdict = f"KEEP [{row['signature'][:50]}] ({kept}/{args.target})"
                if not verdict.startswith("KEEP"):
                    shutil.rmtree(src.parent, ignore_errors=True)
                log(f"{slug:14s} {r['seconds']:5.1f}s {verdict}")
            el = (time.monotonic() - t0) / 60
            eta = (args.target - kept) * el / max(kept, 1)
            log(f"offset {offset}: kept {kept}/{args.target} sigs {len(sig_count)} "
                f"sampled {funnel['sampled']} {el:.0f}min elapsed, ETA ~{eta:.0f}min")
            STATS.write_text(json.dumps({"funnel": funnel, "signatures": sig_count}, indent=1))
    log(f"done: {kept} rows -> {DATA}")


def load_probe(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.load(path.open())
    return {"n": d["n"], "fixed": d["fixed"], "applied": sum(r["applied"] for r in d["results"]),
            "by_id": {r["id"]: r["compiled"] for r in d["results"]}}


def prune(rows: list[dict], funnel: Counter) -> list[dict]:
    """Re-apply the missing-input exclusion (detection was broadened after the run)."""
    keep = []
    for r in rows:
        mf = missing_file({"rawExcerpt": r["first_error"]})
        if mf and not mf.lower().endswith(PKG_EXT):
            funnel["excluded_missing_input"] += 1; funnel["kept"] -= 1
            shutil.rmtree(Path(r["src_dir"]).parent, ignore_errors=True)
            log(f"prune {r['arxiv_id']}: missing input {mf}")
        else:
            r["missing_file"] = mf
            keep.append(r)
    if len(keep) != len(rows):
        DATA.write_text("".join(json.dumps(r) + "\n" for r in keep))
        STATS.write_text(json.dumps({"funnel": funnel, "signatures": Counter(r["signature"] for r in keep)}, indent=1))
    return keep


def report(args: argparse.Namespace) -> None:
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    funnel = Counter(json.load(STATS.open())["funnel"]) if STATS.exists() else Counter()
    rows = prune(rows, funnel)
    sigs = Counter(r["signature"] for r in rows)
    years = Counter(r["year"] for r in rows)
    missing = [r for r in rows if r["missing_file"]]
    md = ["# Organic skew eval (arXiv 2008-2018 under TeX Live 2025)", "",
          f"{len(rows)} rows in `data/skew_eval.jsonl`, sources in `corpus/arxiv_skew/`. No planted "
          "errors: every project fails as-is, so there is no ground-truth patch and `strict`/`exact` "
          "in `run_eval.py` do not apply -- score is compile-only (`fixed`).", "",
          "## Funnel", ""]
    order = ["sampled", "materialized", "compiled_ok", "timed_out", "failed_no_error",
             "compiled_failed", "failed_unlocated", "excluded_missing_input", "capped", "kept"]
    md += [f"- {k}: {funnel.get(k, 0)}" for k in order]
    md += ["", "(`materialized` = single-root, \\documentclass, no psfig/special, 10k-80k chars; "
           "`compiled_failed` = latexmk failed with >= 1 parsed error; `failed_unlocated` = no "
           "error had a source line; `excluded_missing_input` = first error is a missing non-package "
           "file (.tex/.bib/.eps...) that the dataset does not contain -- unfixable, dropped; "
           "`capped` = signature already had 15 rows, or the missing-package quota (30% of target) was full.)", "",
           "## Error signature histogram (first error per project)", "",
           "| n | signature |", "|--:|---|"]
    md += [f"| {n} | `{s}` |" for s, n in sigs.most_common()]
    md += ["", f"{len(sigs)} distinct signatures. By year: " +
           ", ".join(f"{y}: {n}" for y, n in sorted(years.items())), ""]
    md += ["## Missing-package rows", "",
           f"{len(missing)}/{len(rows)} rows ({len(missing) / max(len(rows), 1):.0%}) have a first error "
           "`File ... not found` for a .sty/.cls/.bst that is neither in TeX Live 2025 nor in the "
           "project (`missing_file` field). Kept: the realistic fix is to drop/replace the package, "
           "but some need the author's private style and are effectively unfixable. Filter on "
           "`missing_file` to score without them.", ""]
    if missing:
        md += ["Most common: " + ", ".join(f"`{f}` ({n})" for f, n in Counter(
            r["missing_file"] for r in missing).most_common(8)), ""]
    md += ["## Example error excerpts", ""]
    seen = set()
    for r in sorted(rows, key=lambda r: bool(r["missing_file"])):  # drift bugs before missing classes
        if r["signature"] in seen or len(seen) >= 3:
            continue
        seen.add(r["signature"])
        errs = "\n".join(f"- line {e['line']}: {e['text']}" for e in r["error_lines"][:4])
        src = (Path(r["src_dir"]) / "main.tex").read_text(errors="replace").split("\n")
        ln = r["error_lines"][0]["line"]
        ctx = "\n".join(f"{i + 1}: {src[i]}"[:160] for i in range(max(0, ln - 3), min(len(src), ln + 2)))
        md += [f"### arXiv {r['arxiv_id']} ({r['year']}) -- `{r['signature']}`", "", errs, "",
               "Source around the first error:", "```", ctx, "```", ""]
    md += ["## Probe (first 40 rows, compile-only)", ""]
    probes = {"claude-fable-5.1 (gateway)": load_probe(common.PROJ / "out" / "skew_fable.json"),
              "tuned r2 epoch2 (tinker)": load_probe(common.PROJ / "out" / "skew_tuned_r2.json")}
    def family(r: dict) -> str:
        return ("missing-pkg" if r["missing_file"] else "encoding" if "unicode" in r["signature"]
                else "graphics" if "graphic" in r["signature"] else "other")

    fam_of = {r["id"]: family(r) for r in rows}
    fams = ["missing-pkg", "encoding", "graphics", "other"]
    md += ["| arm | fixed | applied | " + " | ".join(fams) + " |", "|---|--:|--:|" + "--:|" * len(fams)]
    for name, p in probes.items():
        if not p:
            md.append(f"| {name} | (not run) | |" + " |" * len(fams))
            continue
        cells = []
        for fam in fams:
            sub = [v for k, v in p["by_id"].items() if fam_of.get(k) == fam]
            cells.append(f"{sum(sub)}/{len(sub)}")
        md.append(f"| {name} | {p['fixed']}/{p['n']} = {p['fixed'] / p['n']:.0%} | {p['applied']}/{p['n']} | "
                  + " | ".join(cells) + " |")
    md += ["", "Families: missing-pkg = first error is a missing .cls/.sty; encoding = Unicode character "
           "errors; graphics = .eps/.ps extension or BoundingBox errors; other = everything else "
           "(option clashes, redefinitions, deprecated macros, package-order errors)."]
    md += ["", "## Caveats", "",
           "- No ground truth: a 'fix' that deletes content still counts. Compile-only; `refs.json` is not "
           "built for these docs (build_refs.py needs a clean baseline compile, which by construction fails).",
           "- U+FFFD rows: the dataset already replaced undecodable bytes (latin-1 sources without inputenc); "
           "the character is unrecoverable, so the only fix is to drop it -- fixable, but content-lossy.",
           "- Figures are valid 1x1 placeholders (pdf/png/jpg/eps; `.eps` auto-converts via restricted "
           "shell-escape). `.ps` includes and `[dvips]` driver options have no pdflatex path, so the "
           "graphics family is real latex+dvips-era drift, not a placeholder artifact.",
           "- Only the root file (`main.tex`) is editable by the model; errors inside project .sty/.cls "
           "or `\\input` files must be fixed indirectly.",
           "- The signature cap (15) flattens the true distribution; the raw counts before capping are in "
           "`data/skew_stats.json` (`capped` in the funnel).",
           "- Compiles ran under heavy machine load (load avg ~70); `compile_seconds` is not a latency signal.",
           ]
    REPORT.write_text("\n".join(md) + "\n")
    print(REPORT.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=80_000)
    ap.add_argument("--max-per-sig", type=int, default=15)
    ap.add_argument("--max-missing-frac", type=float, default=0.3,
                    help="cap on rows whose first error is a missing .sty/.cls")
    ap.add_argument("--max-sampled", type=int, default=6000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    (report if args.report else generate)(args)


if __name__ == "__main__":
    main()
