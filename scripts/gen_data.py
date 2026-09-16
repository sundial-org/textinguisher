"""Generate (broken doc + real compile log -> fix patch) examples from the template corpus.

Phase 1: baseline-compile every template, keep the clean+fast ones.
Phase 2: doc x mutation x seed -> apply break edit, compile, keep real failures,
build the production fix prompt and the reverse-edit target. Writes JSONL.
"""
import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import common
import perturb


def discover_docs() -> list[dict]:
    docs = []
    for manifest in sorted(common.CORPUS.glob("*/*/manifest.yaml")):
        tdir = manifest.parent
        main = tdir / "files" / "main.tex"
        if not main.exists() or "\\documentclass" not in main.read_text(errors="replace"):
            continue
        m = re.search(r"shared_bundles:\s*\[([^\]]*)\]", manifest.read_text())
        bundles = [b.strip() for b in m.group(1).split(",") if b.strip()] if m else []
        docs.append({
            "slug": f"{tdir.parent.name}/{tdir.name}",
            "src_dir": str(tdir / "files"),
            "bundles": [str(common.CORPUS / "_shared" / b) for b in bundles],
        })
    for main in sorted((common.PROJ / "corpus").glob("arxiv*/*/files/main.tex")):
        docs.append({"slug": f"{main.parent.parent.parent.name}/{main.parent.parent.name}",
                     "src_dir": str(main.parent), "bundles": []})
    return docs


def baseline_one(doc: dict) -> dict:
    r = common.compile_project(Path(doc["src_dir"]), [Path(b) for b in doc["bundles"]], timeout=90)
    return {**doc, "ok": r["ok"], "seconds": r["seconds"], "engine": r["engine"],
            "n_errors": len(r["error_items"])}


def gen_one(doc: dict, mutation: str, seed: int) -> dict | None:
    text = (Path(doc["src_dir"]) / "main.tex").read_text(errors="replace")
    edit = perturb.make_break(text, mutation, seed)
    if not edit:
        return None
    broken = text.replace(edit["search"], edit["replace"], 1)
    r = common.compile_project(
        Path(doc["src_dir"]), [Path(b) for b in doc["bundles"]],
        overrides={"main.tex": broken}, timeout=90,
    )
    if r["ok"]:
        return None  # perturbation didn't actually break the compile
    project_files = {p.name for p in Path(doc["src_dir"]).iterdir()}
    err_lines = common.error_lines_from_items(r["items"], "main.tex", project_files)
    if not err_lines:
        return None  # no located error -> below prod quality bar, skip for v1
    prompt = common.build_fix_prompt("main.tex", err_lines, r["log"], broken)
    target = common.format_edits([(edit["replace"], edit["search"])])
    # sanity: the target must round-trip the broken doc back to the original
    fixed, err = common.apply_edits(broken, common.parse_edits(target))
    if err or fixed != text:
        return None
    return {
        "id": f"{doc['slug'].replace('/', '_')}__{mutation}__{seed}",
        "doc": doc["slug"], "src_dir": doc["src_dir"], "bundles": doc["bundles"],
        "mutation": mutation, "seed": seed, "break_edit": edit,
        "engine": r["engine"], "compile_seconds": r["seconds"],
        "error_lines": err_lines, "log_tail": r["log"][-common.MAX_LOG_CHARS:],
        "prompt": prompt, "target": target,
    }


V2_FULL_LINES = 500
V2_MAX_CHARS = 16000


def sites_visible(prompt: str, edits: list[dict]) -> bool:
    """Every corrupted line must appear in the prompt, or no model can anchor a fix."""
    for e in edits:
        lines = [l.strip() for l in e["replace"].splitlines() if len(l.strip()) >= 8]
        if not lines:
            lines = [l.strip() for l in e["search"].splitlines() if len(l.strip()) >= 8][:1]
        if not all(l[:60] in prompt for l in lines):
            return False
    return True


def gen_one_multi(doc: dict, k: int, seed: int) -> dict | None:
    """v2 hard example: k stacked mutations, full-file context, multi-block target."""
    text = (Path(doc["src_dir"]) / "main.tex").read_text(errors="replace")
    if len(text) > V2_MAX_CHARS * 2:
        return None  # keep the whole file visible in the prompt
    got = perturb.make_break_multi(text, k, seed)
    if not got:
        return None
    broken, edits = got
    r = common.compile_project(
        Path(doc["src_dir"]), [Path(b) for b in doc["bundles"]],
        overrides={"main.tex": broken}, timeout=90,
    )
    if r["ok"]:
        return None
    project_files = {p.name for p in Path(doc["src_dir"]).iterdir()}
    err_lines = common.error_lines_from_items(r["items"], "main.tex", project_files)
    prompt = common.build_fix_prompt("main.tex", err_lines, r["log"], broken,
                                     full_lines=V2_FULL_LINES, max_chars=V2_MAX_CHARS)
    if not sites_visible(prompt, edits):
        return None  # hard-but-fair: never ask for a fix the model cannot see
    fix_blocks = [(e["replace"], e["search"]) for e in reversed(edits)]
    target = common.format_edits(fix_blocks)
    fixed, err = common.apply_edits(broken, common.parse_edits(target))
    if err or fixed != text:
        return None
    return {
        "id": f"{doc['slug'].replace('/', '_')}__multi{len(edits)}__{seed}",
        "doc": doc["slug"], "src_dir": doc["src_dir"], "bundles": doc["bundles"],
        "mutation": f"multi{len(edits)}", "seed": seed,
        "break_edits": edits, "n_breaks": len(edits),
        "engine": r["engine"], "compile_seconds": r["seconds"],
        "error_lines": err_lines, "log_tail": r["log"][-common.MAX_LOG_CHARS:],
        "prompt": prompt, "target": target,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.DATA / "examples.jsonl"))
    ap.add_argument("--per-mutation", type=int, default=3, help="seeds per doc x mutation")
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--max-baseline-seconds", type=float, default=45.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="", help="keep docs whose slug contains this substring")
    ap.add_argument("--compose", type=int, default=0,
                    help="v2 mode: stack 2..N mutations per example, full-file context")
    ap.add_argument("--docs-file", default="", help="JSON list of doc slugs to use (e.g. eval docs)")
    ap.add_argument("--exclude-docs-file", default="", help="JSON list of doc slugs to skip (held-out)")
    ap.add_argument("--seed-offset", type=int, default=0, help="first seed (for extra passes)")
    args = ap.parse_args()

    docs = [d for d in discover_docs() if args.only in d["slug"]]
    if args.docs_file:
        wanted = set(json.loads(Path(args.docs_file).read_text()))
        docs = [d for d in docs if d["slug"] in wanted]
    if args.exclude_docs_file:
        banned = set(json.loads(Path(args.exclude_docs_file).read_text()))
        docs = [d for d in docs if d["slug"] not in banned]
    if args.limit_docs:
        docs = docs[: args.limit_docs]
    print(f"{len(docs)} candidate docs; baseline compiling...")

    good = []
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(baseline_one, d) for d in docs]):
            r = f.result()
            status = "ok" if r["ok"] else f"FAIL({r['n_errors']} errors)"
            print(f"  {r['slug']:40s} {r['engine']:9s} {r['seconds']:6.1f}s  {status}")
            if r["ok"] and r["seconds"] <= args.max_baseline_seconds:
                good.append(r)
    print(f"baseline: {len(good)}/{len(docs)} usable\n")

    if args.compose:
        import random as _random
        _rng = _random.Random(1)
        tasks = [(d, _rng.randint(2, args.compose), s)
                 for d in good for s in range(args.seed_offset, args.seed_offset + args.per_mutation)]
        worker = gen_one_multi
    else:
        tasks = [(d, fn.__name__, s)
                 for d in good for fn in perturb.MUTATIONS for s in range(args.per_mutation)]
        worker = gen_one
    print(f"{len(tasks)} perturbation tasks; compiling...")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n_kept, by_mutation = 0, {}
    with out.open("w") as fh, ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(worker, d, m, s) for d, m, s in tasks]
        for i, f in enumerate(as_completed(futs)):
            row = f.result()
            if row:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n_kept += 1
                by_mutation[row["mutation"]] = by_mutation.get(row["mutation"], 0) + 1
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.monotonic() - t0)
                eta_min = (len(tasks) - i - 1) / rate / 60
                print(f"  {i + 1}/{len(tasks)} done, {n_kept} kept, eta {eta_min:.0f}m")
    print(f"\nwrote {n_kept} examples to {out}")
    for m, n in sorted(by_mutation.items(), key=lambda kv: -kv[1]):
        print(f"  {m:18s} {n}")


if __name__ == "__main__":
    main()
