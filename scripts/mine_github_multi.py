"""Mine REAL multi-file LaTeX compile fixes from GitHub commits (project-lane rows -> data/github_multi.jsonl).

search : gh commit search (30 req/h budget) -> corpus/github_multi/candidates.jsonl
mine   : per commit: `git fetch --depth 2 <sha>`, root detection, diff filters (text files only, <=4 files,
         <=40 lines), materialize the CHILD (fixed) project under corpus/github_multi/<owner>__<repo>__<sha7>/files
         (root renamed main.tex), file_edits = search(fixed)/replace(parent) per file with an exact round trip,
         compile CHILD (must pass) and PARENT (must fail with a located error) -> data/github_multi.jsonl.
         Resumable (corpus/github_multi/processed.jsonl); COMPILE_BACKEND=modal recommended, then:
verify : recompile every kept pair locally (the scoring toolchain), merge PDF refs into data/refs.json,
         write the frozen by-repo split (data/multi_split.json, github_multi_{heldout,train}.jsonl).
         `mine` refuses to run once the split exists (rows appended later would be unverified and unsplit).

  mine_github_multi.py search [--limit 300]      # commit search (keyword)
  mine_github_multi.py scan [--workers 12]       # LaTeX repo search + local commit-message grep (higher precision)
  COMPILE_BACKEND=modal MODAL_PROFILE=sundial-prod mine_github_multi.py mine [--workers 16]
  mine_github_multi.py verify [--workers 8]
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import common
from gen_project import make_row
from mine_texse import derive_edits, normalize_error

GH = common.PROJ / "corpus" / "github_multi"
CANDS, PROCESSED, REPOS = GH / "candidates.jsonl", GH / "processed.jsonl", GH / "repos.json"
OUT = common.DATA / "github_multi.jsonl"
QUERIES = ['"undefined control sequence"', '"fix compile error" latex', '"latex error"', '"missing $ inserted"',
           '"fix latexmk"', '"fix build" tex', '"compile error" tex', '"fix compilation" latex', '"runaway argument"',
           '"extra alignment tab"', '"file ended while scanning"', '"does not compile" latex', '"compiles now" latex',
           '"fix compile" thesis', '"missing \\begin{document}"', '"paragraph ended before"', '"fix latex build"',
           '"fix compile" paper', '"undefined control sequence" fix', '"latex compile" fix']
BUILD_EXTS = {".aux", ".log", ".out", ".toc", ".blg", ".fls", ".fdb_latexmk", ".synctex.gz", ".nav", ".snm", ".lof",
              ".lot", ".bcf", ".run.xml", ".xdv", ".dvi"}
IGNORE_EXTS = {".md", ".yml", ".yaml", ".json", ".toml", ".py", ".sh", ".rst", ".html"}
SRC_EXTS = {".tex", ".sty", ".cls", ".bib"}
SHELL_RE = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\b(?:minted|pythontex|gnuplottex|svg)\b|\\write18|-shell-escape")
DEP_RE = re.compile(r"\\(?:input|include|subfile|InputIfFileExists|import\{[^}]*\})\s*\{([^{}]+)\}")
MAX_REPO_KB, MAX_PROJ_BYTES, MAX_PROJ_FILES, MAX_PER_REPO = 2_000_000, 80_000_000, 1500, 5


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def sh(args, cwd=None, timeout=300) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, timeout=timeout)


def text_of(b: bytes) -> str:
    return b.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


# --- search ---

def search(args) -> None:
    seen = {json.loads(l)["sha"] for l in CANDS.read_text().splitlines()} if CANDS.exists() else set()
    GH.mkdir(parents=True, exist_ok=True)
    with CANDS.open("a") as fh:
        for q in QUERIES[args.skip:]:
            while True:
                rem = json.loads(sh(["gh", "api", "rate_limit"]).stdout)["resources"]["search"]
                need = -(-args.limit // 100)
                if rem["remaining"] >= need:
                    break
                log(f"search budget {rem['remaining']} < {need}; sleeping until reset")
                time.sleep(max(5, rem["reset"] - time.time() + 5))
            new = 0
            for page in range(1, need + 1):  # one page per call, paced: back-to-back pages trip the secondary limit
                for attempt in range(4):
                    p = sh(["gh", "api", "-X", "GET", "search/commits", "-f", f"q={q}", "-f", "per_page=100", "-f", f"page={page}"])
                    if p.returncode == 0:
                        break
                    log(f"query {q!r} p{page} failed: {p.stderr.decode()[:100]}"); time.sleep(120 * (attempt + 1))
                if p.returncode:
                    break
                items = json.loads(p.stdout).get("items", [])
                for r in items:
                    if len(r["parents"]) != 1 or r["sha"] in seen:
                        continue
                    seen.add(r["sha"]); new += 1
                    fh.write(json.dumps({"repo": r["repository"]["full_name"], "sha": r["sha"], "parent": r["parents"][0]["sha"],
                                         "message": r["commit"]["message"], "url": r["html_url"], "query": q}) + "\n")
                fh.flush(); time.sleep(20)
                if len(items) < 100:
                    break
            log(f"{q!r}: {new} new candidates ({len(seen)} total)"); time.sleep(30)


REPO_QUERIES = ["language:TeX pushed:>2025-06-01", "language:TeX pushed:2025-01-01..2025-06-01", "language:TeX pushed:2024-06-01..2025-01-01",
                "language:TeX stars:>15", "language:TeX thesis", "language:TeX paper", "language:TeX dissertation", "language:TeX notes",
                "language:TeX lecture", "language:TeX book", "language:TeX template", "language:TeX beamer", "language:TeX arxiv", "language:TeX report",
                # round 2 (--rescan): more windows / topics
                "language:TeX pushed:2024-01-01..2024-06-01", "language:TeX pushed:2023-06-01..2024-01-01", "language:TeX pushed:2023-01-01..2023-06-01",
                "language:TeX stars:5..15", "language:TeX manuscript", "language:TeX journal", "language:TeX proposal", "language:TeX course",
                "language:TeX homework", "language:TeX slides", "language:TeX physics", "language:TeX math", "language:TeX cv",
                "language:TeX ieee", "language:TeX acm", "language:TeX neurips", "language:TeX overleaf", "language:TeX preprint",
                "language:TeX survey", "language:TeX tutorial", "language:TeX exam"]
# broad on purpose: the parent/child compile check is the judge, the message only prunes content-only commits
MSG_RE = r"fix|typo|bug|broken|error|compil|build|repair|correct|missing|undefined|escape|brace|bracket|syntax"  # plain ERE: no \b/\w on macOS


def scan(args) -> None:
    """Repo search (LaTeX repos) -> blobless clone -> grep commit messages locally -> candidates (query='scan')."""
    seen = {json.loads(l)["sha"] for l in CANDS.read_text().splitlines()} if CANDS.exists() else set()
    scanned_path = GH / "scanned_repos.txt"
    scanned = set(scanned_path.read_text().split()) if scanned_path.exists() and not args.rescan else set()
    repo_list = GH / "repo_list.json"  # cached so a --rescan does not repeat the search stage
    repos = json.loads(repo_list.read_text()) if repo_list.exists() else {}
    for q in ([] if repos else REPO_QUERIES):
        for page in range(1, 11):
            p = sh(["gh", "api", "-X", "GET", "search/repositories", "-f", f"q={q}", "-f", "sort=updated", "-f", "per_page=100", "-f", f"page={page}"])
            if p.returncode:
                log(f"repo query {q!r} p{page} failed: {p.stderr.decode()[:100]}"); time.sleep(60); continue
            items = json.loads(p.stdout).get("items", [])
            for r in items:
                if not r["fork"] and r["size"] <= MAX_REPO_KB:
                    repos[r["full_name"]] = r["size"]
            time.sleep(3)
            if len(items) < 100:
                break
        log(f"{q!r}: {len(repos)} repos to scan so far")
    repo_list.write_text(json.dumps(repos))
    repos = {r: sz for r, sz in repos.items() if r not in scanned}

    def scan_repo(repo: str) -> list[dict]:
        d = GH / "_scan" / repo.replace("/", "__")
        if not d.exists():
            try:
                if sh(["git", "clone", "-q", "--bare", "--filter=blob:none", f"https://github.com/{repo}", str(d)], timeout=300).returncode:
                    return []
            except subprocess.TimeoutExpired:
                return []
        out = text_of(sh(["git", "log", "--no-merges", "-i", "-E", f"--grep={MSG_RE}", "--format=%H %P %s", "-n", "40"], cwd=d).stdout)
        cands = []
        for ln in out.splitlines():
            sha, parent, _, msg = ln.split(" ", 3) if ln.count(" ") >= 3 else (ln.split(" ") + ["", "", ""])[:4]
            if len(parent) != 40:
                continue
            ns = text_of(sh(["git", "diff-tree", "--no-commit-id", "--name-status", "-r", "--no-renames", sha], cwd=d).stdout).splitlines()
            tex = [l for l in ns if l.endswith(tuple(common.TEXT_EXTS))]
            if 1 <= len(tex) <= 4 and all(l.startswith("M\t") for l in tex):
                cands.append({"repo": repo, "sha": sha, "parent": parent, "message": msg, "url": f"https://github.com/{repo}/commit/{sha}", "query": "scan"})
        shutil.rmtree(d, ignore_errors=True)
        return cands

    n, t0 = 0, time.monotonic()
    with ThreadPoolExecutor(args.workers) as ex, CANDS.open("a") as fh, scanned_path.open("a") as sf:
        futs = {ex.submit(scan_repo, r): r for r in repos}
        for i, f in enumerate(as_completed(futs)):
            for c in f.result():
                if c["sha"] not in seen:
                    seen.add(c["sha"]); n += 1; fh.write(json.dumps(c) + "\n")
            fh.flush(); sf.write(futs[f] + "\n"); sf.flush()
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.monotonic() - t0)
                log(f"scanned {i + 1}/{len(repos)} repos, {n} new candidates, eta {(len(repos) - i - 1) / rate / 60:.0f}m")
    log(f"scan done: {n} new candidates from {len(repos)} repos")


# --- mine ---

def repo_meta(repo: str, cache: dict, lock) -> dict | None:
    with lock:
        if repo in cache:
            return cache[repo]
    p = sh(["gh", "api", f"repos/{repo}", "--jq",
            '{license: .license.spdx_id, size: .size, stars: .stargazers_count, fork: .fork, pushed: .pushed_at, url: .html_url}'])
    meta = json.loads(p.stdout) if p.returncode == 0 else None
    with lock:
        cache[repo] = meta
        REPOS.write_text(json.dumps(cache, indent=0))
    return meta


def fetch(repo: str, sha: str) -> Path | None:
    d = GH / "_repos" / repo.replace("/", "__")
    if not (d / ".git").exists():
        d.mkdir(parents=True, exist_ok=True)
        sh(["git", "init", "-q"], cwd=d); sh(["git", "remote", "add", "origin", f"https://github.com/{repo}"], cwd=d)
    try:
        p = sh(["git", "fetch", "-q", "--depth", "2", "origin", sha], cwd=d, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    return d if p.returncode == 0 and sh(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=d).returncode == 0 else None


def strip_comments(t: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in t.splitlines())


def closure(root: Path, top: Path) -> tuple[set[Path], str]:
    """Files transitively \\input by root (paths resolved from the root dir, then repo top) + their text."""
    seen, todo, text = set(), [root.resolve()], ""  # resolved like the changed paths (macOS /var -> /private/var)
    while todo:
        f = todo.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        t = strip_comments(f.read_text(errors="replace")); text += t
        for m in DEP_RE.finditer(t):
            rel = m.group(1).strip()
            for base in (root.parent, top):
                for cand in (base / rel, base / (rel + ".tex")):
                    if cand.is_file():
                        todo.append(cand.resolve())
    return seen, text


def pick_root(top: Path, changed: list[str]) -> Path | None:
    texs = [p for p in top.rglob("*.tex") if p.is_file() and p.stat().st_size < 400_000]
    if len(texs) > 300:
        return None
    roots = [p for p in texs if re.search(r"^\s*\\documentclass", p.read_text(errors="replace"), re.M)
             and "\\begin{document}" in p.read_text(errors="replace")]
    if len(roots) == 1:
        return roots[0]
    hits = []
    for r in roots:
        files, text = closure(r, top)
        stems = {Path(c).stem for c in changed}
        if any((top / c).resolve() in files for c in changed) or any(re.search(r"\\(?:usepackage|documentclass|bibliography|addbibresource)\b[^\n]*\b" + re.escape(s) + r"\b", text) for s in stems):
            hits.append(r)
    return hits[0] if len(hits) == 1 else None


def build_files(log_text: str, files: set[str]) -> set[str]:
    """Project files actually opened by TeX (`(./path` tokens; newlines dropped because TeX wraps long lines)."""
    flat = log_text.replace("\n", "")
    return {f for f in files if f"(./{f}" in flat or f"({f}" in flat}


def process(c: dict, meta: dict) -> dict:
    """One candidate commit -> {"stage": ..., "row": ...}."""
    repo, sha, parent = c["repo"], c["sha"], c["parent"]
    if meta["size"] > MAX_REPO_KB:
        return {"stage": "repo_too_big"}
    d = fetch(repo, sha)
    if not d:
        return {"stage": "fetch_failed"}
    if sh(["git", "cat-file", "-e", f"{parent}^{{commit}}"], cwd=d).returncode:
        return {"stage": "fetch_failed"}
    ns = text_of(sh(["git", "diff", "--name-status", "--no-renames", parent, sha], cwd=d).stdout).splitlines()
    changed = []
    for ln in ns:
        st, _, path = ln.partition("\t")
        if Path(path).suffix in BUILD_EXTS | IGNORE_EXTS or path.endswith(".pdf") or Path(path).name.startswith(".") \
                or Path(path).name in ("Makefile", "LICENSE"):
            continue  # committed build outputs / docs / CI / dotfiles are ignored, not diffed
        if st != "M" or Path(path).suffix not in common.TEXT_EXTS:
            return {"stage": "non_text_change"}
        changed.append(path)
    if not changed or len(changed) > 4:
        return {"stage": "too_many_files" if changed else "no_text_change"}
    numstat = text_of(sh(["git", "diff", "--numstat", parent, sha, "--"] + changed, cwd=d).stdout)
    if sum(int(a) + int(b) for a, b, _ in (l.split("\t") for l in numstat.splitlines() if l) if a != "-") > 40:
        return {"stage": "diff_too_big"}
    with tempfile.TemporaryDirectory() as td:
        top = Path(td)
        tar = subprocess.Popen(["git", "archive", sha], cwd=d, stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", td], stdin=tar.stdout, capture_output=True); tar.wait()
        root = pick_root(top, changed)
        if not root:
            return {"stage": "no_single_root"}
        proj = root.parent
        rels = []
        for ch in changed:
            try:
                rels.append(str((top / ch).relative_to(proj)))
            except ValueError:
                return {"stage": "change_outside_project"}
        if (proj / "main.tex").exists() and root.name != "main.tex":
            return {"stage": "main_tex_clash"}
        allf = [p for p in proj.rglob("*") if p.is_file()]
        if len(allf) > MAX_PROJ_FILES or sum(p.stat().st_size for p in allf) > MAX_PROJ_BYTES:
            return {"stage": "project_too_big"}
        slug = f"{repo.replace('/', '__')}__{sha[:7]}"
        src = GH / slug / "files"
        shutil.rmtree(src.parent, ignore_errors=True); src.mkdir(parents=True)
        stem = root.stem
        for p in allf:
            rel = p.relative_to(proj)
            if p.suffix in BUILD_EXTS or (p.parent == proj and p.name == stem + ".pdf"):
                continue
            dst = src / ("main.bbl" if (rel == Path(stem + ".bbl")) else "main.tex" if p == root else rel)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
    files = common.project_files(src)
    if any(SHELL_RE.search(t) for t in files.values()):
        shutil.rmtree(src.parent); return {"stage": "shell_escape"}
    if common.pick_engine(files["main.tex"]) != common.pick_engine("\n".join(files.values())):
        shutil.rmtree(src.parent); return {"stage": "engine_in_child"}
    edits, why = [], "identical"
    for ch, rel in zip(changed, rels):
        rel = "main.tex" if rel == str(root.relative_to(proj)) else rel
        old = text_of(sh(["git", "show", f"{parent}:{ch}"], cwd=d).stdout)
        if rel not in files:
            shutil.rmtree(src.parent); return {"stage": "non_text_change"}
        es, why = derive_edits(files[rel], old)
        if es is None and why != "identical":
            shutil.rmtree(src.parent); return {"stage": why}
        edits += [(rel, e["search"], e["replace"], "real") for e in (es or [])]
    if not edits:
        shutil.rmtree(src.parent); return {"stage": "identical"}
    if sum(len(s.split("\n")) + len(r.split("\n")) for _, s, r, _ in edits) > 80:
        shutil.rmtree(src.parent); return {"stage": "diff_too_big"}
    broken = dict(files)
    for f, s, r, _ in edits:
        broken[f] = broken[f].replace(s, r, 1)
    res = check_pair(src, files, broken, edits)
    if res["stage"] != "kept":
        shutil.rmtree(src.parent); return res
    row = res["row"]
    row.update({
        "id": f"github_{slug}", "doc": f"github_multi/{slug}", "src_dir": str(src), "mutation": "real", "seed": 0,
        "source": "github", "repo": repo, "sha": sha, "parent_sha": parent, "license": meta["license"], "stars": meta["stars"],
        "url": c["url"], "commit_message": c["message"][:2000], "changed_files": rels, "original_root": str(root.relative_to(proj)),
        "n_files": sum(1 for f in files if Path(f).suffix in SRC_EXTS), "engine": res["engine"],
        "error_category": normalize_error(row["error_lines"][0]["text"]), "query": c.get("query", ""),
    })
    return {"stage": "kept", "row": row}


def check_pair(src: Path, files: dict, broken: dict, edits: list, timeout: int = 90) -> dict:
    """Child must compile (plain, else with bib); parent (child + edits) must fail with a located error."""
    rc, bib = None, False
    for bib in (False, True):
        rc = common.compile_project(src, [], timeout=timeout, bib=bib)
        if rc["ok"]:
            break
    if not rc["ok"]:
        return {"stage": "child_timeout" if rc["timed_out"] else "child_compile_fail"}
    if len(build_files(rc["log"], set(files))) < 2:
        return {"stage": "single_file_build"}
    overrides = {f: broken[f] for f in broken if broken[f] != files[f]}
    rb = common.compile_project(src, [], overrides=overrides, timeout=timeout, bib=bib)
    if rb["timed_out"]:
        return {"stage": "parent_timeout"}
    if rb["ok"]:
        return {"stage": "parent_compiles"}
    row = make_row(src, files, broken, edits, rb, {"bib": bib})
    if not row or not row["error_lines"]:
        return {"stage": "parent_no_located_error"}
    m = re.search(r"File `([^']+)' not found|I can't find file `([^']+)'", row["error_lines"][0]["text"])
    if m:
        missing = Path((m.group(1) or m.group(2)).strip()).stem
        if not any(Path(f).stem == missing for f in (str(p.relative_to(src)) for p in src.rglob("*"))):
            return {"stage": "missing_asset"}
    return {"stage": "kept", "row": row, "engine": rc["engine"], "ref": {"text": rc["pdf_text"], "pages": rc["pdf_pages"]}}


def mine(args) -> None:
    assert not (common.DATA / "multi_split.json").exists(), "multi_split.json is frozen; delete it and re-run verify after mining"
    cands = [json.loads(l) for l in CANDS.read_text().splitlines()]
    retry = set(args.retry.split(",")) if args.retry else set()
    done = {p["sha"] for p in map(json.loads, PROCESSED.read_text().splitlines()) if p["stage"] not in retry} if PROCESSED.exists() else set()
    cache = json.loads(REPOS.read_text()) if REPOS.exists() else {}
    per_repo = Counter(json.loads(l)["repo"] for l in OUT.read_text().splitlines()) if OUT.exists() else Counter()
    by_repo: dict[str, list] = {}
    for c in cands:
        if c["sha"] not in done:
            by_repo.setdefault(c["repo"], []).append(c)
    import threading
    lock = threading.Lock()
    stats, t0, n_done, total = Counter(), time.monotonic(), 0, sum(len(v) for v in by_repo.values())
    log(f"{total} candidates in {len(by_repo)} repos to process ({len(done)} done before)")

    def run_repo(repo: str, cs: list) -> list:
        out = []
        meta = repo_meta(repo, cache, lock)
        for c in cs:
            if meta is None or meta["fork"]:
                out.append((c, {"stage": "repo_unavailable" if meta is None else "fork"})); continue
            if per_repo[repo] >= MAX_PER_REPO:
                out.append((c, {"stage": "repo_cap"})); continue
            try:
                res = process(c, meta)
            except Exception as e:  # noqa: BLE001
                res = {"stage": "worker_error", "err": repr(e)[:200]}
            if res["stage"] == "kept":
                with lock:
                    per_repo[repo] += 1
            out.append((c, res))
        return out

    with ThreadPoolExecutor(args.workers) as ex, PROCESSED.open("a") as pf, OUT.open("a") as of:
        futs = [ex.submit(run_repo, r, cs) for r, cs in by_repo.items()]
        for f in as_completed(futs):
            for c, res in f.result():
                stats[res["stage"]] += 1; n_done += 1
                pf.write(json.dumps({"sha": c["sha"], "repo": c["repo"], "stage": res["stage"], "err": res.get("err")}) + "\n")
                if res["stage"] == "kept":
                    of.write(json.dumps(res["row"]) + "\n")
                    log(f"  KEPT {c['repo']}@{c['sha'][:7]} [{res['row']['error_category']}] fix={res['row']['fix_files']} reported={res['row']['reported_file']}")
            pf.flush(); of.flush()
            rate = n_done / (time.monotonic() - t0)
            log(f"{n_done}/{total} kept={stats['kept']} eta {(total - n_done) / max(rate, 1e-6) / 60:.0f}m  {dict(stats.most_common(6))}")
    shutil.rmtree(GH / "_repos", ignore_errors=True) if args.clean else None
    (GH / "mine_stats.json").write_text(json.dumps(dict(stats), indent=1))
    log(f"done: {dict(stats)}")


# --- verify locally + split ---

def verify_one(row: dict) -> tuple[dict, dict | None]:
    src = Path(row["src_dir"])
    files = common.project_files(src)
    broken = dict(files)
    for e in row["file_edits"]:
        broken[e["file"]] = broken[e["file"]].replace(e["search"], e["replace"], 1)
    res = check_pair(src, files, broken, [(e["file"], e["search"], e["replace"], e["type"]) for e in row["file_edits"]], timeout=180)
    if res["stage"] != "kept":
        return row, {"stage": res["stage"]}
    new = {**row, **res["row"], "bib": res["row"]["bib"], "engine": res["engine"],
           "error_category": normalize_error(res["row"]["error_lines"][0]["text"])}
    return new, {"stage": "kept", "ref": res["ref"]}


def verify(args) -> None:
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    refs_path = common.DATA / "refs.json"
    refs = json.loads(refs_path.read_text()) if refs_path.exists() else {}
    kept, stats = [], Counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(verify_one, r) for r in rows]):
            row, res = f.result()
            stats[res["stage"]] += 1
            if res["stage"] == "kept":
                kept.append(row); refs[row["doc"]] = res["ref"]
            else:
                log(f"  DROP {row['id']}: {res['stage']}"); shutil.rmtree(Path(row["src_dir"]).parent, ignore_errors=True)
    kept.sort(key=lambda r: r["id"])
    OUT.write_text("".join(json.dumps(r) + "\n" for r in kept))
    refs_path.write_text(json.dumps(refs))
    repos = sorted({r["repo"] for r in kept})
    held = {r for r in repos if int(hashlib.sha1(r.encode()).hexdigest(), 16) % 100 < args.heldout_pct}
    (common.DATA / "multi_split.json").write_text(json.dumps({"heldout_repos": sorted(held), "train_repos": sorted(set(repos) - held)}, indent=1))
    for name, pred in (("heldout", lambda r: r["repo"] in held), ("train", lambda r: r["repo"] not in held)):
        (common.DATA / f"github_multi_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept if pred(r)))
    synth = common.DATA / "multi_synth.jsonl"
    if synth.exists():  # synthetic split: held-out docs = project_eval's eval docs (already held out from training)
        ev = set(json.loads((common.DATA / "project_eval_docs.json").read_text()))
        srows = [json.loads(l) for l in synth.read_text().splitlines()]
        for name, pred in (("heldout", lambda r: r["doc"] in ev), ("train", lambda r: r["doc"] not in ev)):
            (common.DATA / f"multi_synth_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in srows if pred(r)))
        split = json.loads((common.DATA / "multi_split.json").read_text())
        split["synth_heldout_docs"] = sorted(ev & {r["doc"] for r in srows})
        (common.DATA / "multi_split.json").write_text(json.dumps(split, indent=1))
    log(f"verified {stats}; {len(kept)} rows, {len(repos)} repos, heldout {sum(r['repo'] in held for r in kept)} rows / {len(held)} repos")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "scan", "mine", "verify"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--skip", type=int, default=0, help="search: skip the first N queries")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--heldout-pct", type=int, default=50)
    ap.add_argument("--clean", action="store_true", help="mine: delete cloned repos afterwards")
    ap.add_argument("--rescan", action="store_true", help="scan: re-clone repos scanned before (e.g. after widening MSG_RE)")
    ap.add_argument("--retry", default="", help="mine: comma-separated stages to reprocess (e.g. after a backend change)")
    args = ap.parse_args()
    {"search": search, "scan": scan, "mine": mine, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    main()
