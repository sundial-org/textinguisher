"""Compiler-found LaTeX fix pairs: enumerate LaTeX repos, walk their histories, compile every tex-touching commit
that passes the diff filter, keep parent-fails -> child-passes transitions. Reuses mine_github_multi as a library.

  mine_github_multi2.py repos                 # -> corpus/github_multi2/repos.jsonl (latex-action CI repos, language:TeX slices, old scan list)
  COMPILE_BACKEND=modal MODAL_PROFILE=sundial-prod mine_github_multi2.py walk [--limit 10] [--hours 5] [--workers 48]
Refs (corpus/github_multi2/refs.jsonl) come from the farm's child compile; finish_multi2.py merges them into data/refs.json
and freezes the split.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import common
import mine_github_multi as m1
from gen_project import make_row
from mine_texse import derive_edits, normalize_error

G2 = common.PROJ / "corpus" / "github_multi2"
REPOS, DONE, STATS, REFS = G2 / "repos.jsonl", G2 / "done.jsonl", G2 / "stats.json", G2 / "refs.jsonl"
OUT, OUT1, LOG = common.DATA / "github_multi2.jsonl", common.DATA / "github_single.jsonl", common.PROJ / "out" / "mine_github_multi2.log"
MAX_REPO_KB, MAX_PROJ_BYTES, MAX_COMMITS, MAX_KEPT, DEPTH, TIMEOUT = 200_000, 50_000_000, 60, 5, 300, 60
LOCAL = threading.Semaphore(int(os.environ.get("LOCAL_SLOTS", "5")))  # hybrid: local latexmk when a slot is free, else the farm
log = m1.log


def sh(args, cwd=None, timeout=300):
    """git with retries: lazy blob fetches from the blobless clone fail transiently under network load."""
    for attempt in range(4):
        p = m1.sh(args, cwd=cwd, timeout=timeout)
        if p.returncode == 0 or not any(k in p.stderr for k in (b"resolve host", b"promisor", b"Could not read", b"early EOF", b"unable to access")):
            return p
        time.sleep(10 * (attempt + 1))
    return p


def logf(msg: str) -> None:
    log(msg)
    with LOG.open("a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# --- repos ---

def paged(endpoint: str, q: str, pause: float, **kw) -> list[dict]:
    items = []
    for page in range(1, 11):
        for attempt in range(3):
            p = sh(["gh", "api", "-X", "GET", endpoint, "-f", f"q={q}", "-f", "per_page=100", "-f", f"page={page}"]
                   + sum((["-f", f"{k}={v}"] for k, v in kw.items()), []))
            if p.returncode == 0:
                break
            log(f"{endpoint} {q!r} p{page}: {p.stderr.decode()[:80]}"); time.sleep(60)
        time.sleep(pause)
        if p.returncode:
            break
        got = json.loads(p.stdout).get("items", [])
        items += got
        if len(got) < 100:
            break
    log(f"{q!r}: {len(items)} items")
    return items


def graphql_meta(names: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(names), 100):
        chunk = names[i:i + 100]
        q = "{" + " ".join(f'r{j}: repository(owner:"{n.split("/")[0]}", name:"{n.split("/")[1]}")'
                           "{isFork diskUsage stargazerCount licenseInfo{spdxId} defaultBranchRef{name}}" for j, n in enumerate(chunk)) + "}"
        p = sh(["gh", "api", "graphql", "-f", f"query={q}"])
        try:
            data = json.loads(p.stdout).get("data") or {}
        except json.JSONDecodeError:
            data = {}
        for j, n in enumerate(chunk):
            r = data.get(f"r{j}")
            if r:
                out[n] = {"repo": n, "fork": r["isFork"], "size": r["diskUsage"], "stars": r["stargazerCount"],
                          "license": (r["licenseInfo"] or {}).get("spdxId"), "default_branch": (r["defaultBranchRef"] or {}).get("name")}
        if (i // 100) % 20 == 0:
            log(f"graphql meta {i + len(chunk)}/{len(names)}")
        time.sleep(1)
    return out


def repos(args) -> None:
    G2.mkdir(parents=True, exist_ok=True)
    have = {r["repo"]: r for r in read_jsonl(REPOS)}
    src: dict[str, set] = {}

    def add(name: str, source: str):
        src.setdefault(name, set()).add(source)

    for n in (common.PROJ / "corpus" / "github_multi" / "scanned_repos.txt").read_text().split():
        add(n, "scan1")
    for action in ("xu-cheng/latex-action", "dante-ev/latex-action"):  # CI repos: histories tend to compile
        for size in ("<600", "600..900", "900..1200", "1200..1600", "1600..2500", ">2500"):
            for it in paged("search/code", f"{action} path:.github/workflows size:{size}", 7):
                add(it["repository"]["full_name"], "ci")
    meta = {}
    months = [f"{y}-{m:02d}-01" for y in (2025, 2026) for m in range(1, 13)]
    windows = [f"{a}..{b}" for a, b in zip(months, months[1:]) if a >= "2025-06-01" and b <= "2026-10-01"]
    for q in [f"language:TeX pushed:{w}" for w in windows] + ["language:TeX stars:>50", "language:TeX stars:20..50", "language:TeX stars:8..20"]:
        for it in paged("search/repositories", q, 2.5, sort="updated"):
            meta[it["full_name"]] = {"repo": it["full_name"], "fork": it["fork"], "size": it["size"], "stars": it["stargazers_count"],
                                     "license": (it["license"] or {}).get("spdx_id"), "default_branch": it["default_branch"]}
            add(it["full_name"], "search")
    need = [n for n in src if n not in meta and n not in have]
    log(f"{len(src)} repo names, {len(need)} need metadata via graphql")
    meta.update(graphql_meta(need))
    with REPOS.open("a") as fh:
        n = 0
        for name in src:
            m = meta.get(name) or have.get(name)
            if name in have or not m or m["fork"] or m["size"] > MAX_REPO_KB or m.get("default_branch") is None:
                continue
            fh.write(json.dumps({**m, "sources": sorted(src[name])}) + "\n"); n += 1
    log(f"wrote {n} new repos to {REPOS} ({len(have) + n} total)")


# --- walk ---

def pick_root(top: Path, changed: list[str]) -> Path | None:
    """m1.pick_root, but an ambiguous set of roots that all include a changed file resolves to the largest document."""
    root = m1.pick_root(top, changed)
    if root:
        return root
    texs = [p for p in top.rglob("*.tex") if p.is_file() and p.stat().st_size < 400_000]
    if len(texs) > 300:
        return None
    hits = []
    for r in texs:
        t = r.read_text(errors="replace")
        if m1.re.search(r"^\s*\\documentclass", t, m1.re.M) and "\\begin{document}" in t:
            files, _ = m1.closure(r, top)
            if any((top / c).resolve() in files for c in changed):
                hits.append((len(files), r))
    return max(hits)[1] if hits else None


class Walker:
    def __init__(self, repo: dict, seen: set, deadline: float):
        self.r, self.repo, self.seen, self.deadline = repo, repo["repo"], seen, deadline
        self.cache: dict[tuple[str, bool], dict] = {}
        self.stages, self.rows, self.refs, self.errs, self.n_compiles, self.n_local, self.compile_secs = Counter(), [], {}, [], 0, 0, 0.0
        self.d = G2 / "_repos" / self.repo.replace("/", "__")

    def compile(self, sha: str, src: Path, bib: bool, overrides=None) -> dict:
        key = (sha, bib, str(src), hashlib.sha1(json.dumps(overrides or {}, sort_keys=True).encode()).hexdigest())
        if key not in self.cache:  # a sha is compiled clean as a child and with overrides as the next pair's parent
            self.n_compiles += 1
            local = LOCAL.acquire(blocking=False)
            try:
                self.cache[key] = common.compile_project(src, [], overrides=overrides, timeout=TIMEOUT, bib=bib, backend="local" if local else None)
            finally:
                LOCAL.release() if local else None
            self.compile_secs += self.cache[key]["seconds"]; self.n_local += local
        return self.cache[key]

    def commits(self) -> list[tuple[str, str, str]]:
        shutil.rmtree(self.d, ignore_errors=True); self.d.parent.mkdir(parents=True, exist_ok=True)
        try:
            p = sh(["git", "clone", "-q", "--filter=blob:none", "--depth", str(DEPTH), "--single-branch", "--no-tags",
                    f"https://github.com/{self.repo}", str(self.d)], timeout=300)
        except subprocess.TimeoutExpired:
            return []
        if p.returncode:
            return []
        out = m1.text_of(sh(["git", "log", "--no-merges", "--format=%H %P\t%s", "--", "*.tex", "*.sty", "*.cls", "*.bib"], cwd=self.d).stdout)
        cs = []
        for ln in out.splitlines():
            head, _, msg = ln.partition("\t")
            parts = head.split()
            if len(parts) == 2 and (self.repo, parts[0]) not in self.seen:
                cs.append((parts[0], parts[1], msg))
        return cs[:MAX_COMMITS]

    def diff_filter(self, sha: str, parent: str):
        if sh(["git", "cat-file", "-e", f"{parent}^{{commit}}"], cwd=self.d).returncode:
            return "parent_missing"
        ns = m1.text_of(sh(["git", "diff", "--name-status", "--no-renames", parent, sha], cwd=self.d).stdout).splitlines()
        changed = []
        for ln in ns:
            st, _, path = ln.partition("\t")
            pp = Path(path)
            if pp.suffix in m1.BUILD_EXTS | m1.IGNORE_EXTS or path.endswith(".pdf") or pp.name.startswith(".") or pp.name in ("Makefile", "LICENSE"):
                continue
            if st != "M" or pp.suffix not in common.TEXT_EXTS:
                return "non_text_change"
            changed.append(path)
        if not changed or len(changed) > 4:
            return "too_many_files" if changed else "no_text_change"
        numstat = m1.text_of(sh(["git", "diff", "--numstat", parent, sha, "--"] + changed, cwd=self.d).stdout)
        if sum(int(a) + int(b) for a, b, _ in (l.split("\t") for l in numstat.splitlines() if l) if a != "-") > 40:
            return "diff_too_big"
        return changed

    def pair(self, sha: str, parent: str, msg: str, changed: list[str]) -> dict:
        d = self.d
        with tempfile.TemporaryDirectory() as td:
            top = Path(td)
            ar = sh(["git", "archive", sha], cwd=d)
            if ar.returncode:
                return {"stage": "archive_failed"}
            subprocess.run(["tar", "-x", "-C", td], input=ar.stdout, capture_output=True)
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
            if len(allf) > m1.MAX_PROJ_FILES or sum(p.stat().st_size for p in allf) > MAX_PROJ_BYTES:
                return {"stage": "project_too_big"}
            slug = f"{self.repo.replace('/', '__')}__{sha[:7]}"
            src = G2 / slug / "files"
            shutil.rmtree(src.parent, ignore_errors=True); src.mkdir(parents=True)
            stem = root.stem
            for p in allf:
                rel = p.relative_to(proj)
                if p.suffix in m1.BUILD_EXTS or (p.parent == proj and p.name == stem + ".pdf"):
                    continue
                dst = src / ("main.bbl" if rel == Path(stem + ".bbl") else "main.tex" if p == root else rel)
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
        try:
            res = self.check(sha, parent, src, root, proj, changed, rels)
        finally:
            if res.get("stage") != "kept":
                shutil.rmtree(src.parent, ignore_errors=True)
        if res["stage"] != "kept":
            return res
        row = res["row"]
        row.update({
            "id": f"github2_{slug}", "doc": f"github_multi2/{slug}", "src_dir": str(src), "mutation": "real", "seed": 0,
            "source": "github", "repo": self.repo, "sha": sha, "parent_sha": parent, "license": self.r["license"], "stars": self.r["stars"],
            "url": f"https://github.com/{self.repo}/commit/{sha}", "commit_message": msg[:2000], "changed_files": rels,
            "original_root": str(root.relative_to(proj)), "n_files": sum(1 for f in res["files"] if Path(f).suffix in m1.SRC_EXTS),
            "engine": res["engine"], "error_category": normalize_error(row["error_lines"][0]["text"]), "query": "walk",
            "single_file_build": res["single"],
        })
        return {"stage": "kept", "row": row, "ref": res["ref"]}

    def check(self, sha, parent, src, root, proj, changed, rels) -> dict:
        files = common.project_files(src)
        if any(m1.SHELL_RE.search(t) for t in files.values()):
            return {"stage": "shell_escape"}
        if common.pick_engine(files["main.tex"]) != common.pick_engine("\n".join(files.values())):
            return {"stage": "engine_in_child"}
        edits = []
        for ch, rel in zip(changed, rels):
            rel = "main.tex" if rel == str(root.relative_to(proj)) else rel
            if rel not in files:
                return {"stage": "non_text_change"}
            old = m1.text_of(sh(["git", "show", f"{parent}:{ch}"], cwd=self.d).stdout)
            es, why = derive_edits(files[rel], old)
            if es is None and why != "identical":
                return {"stage": why}
            edits += [(rel, e["search"], e["replace"], "real") for e in (es or [])]
        if not edits:
            return {"stage": "identical"}
        if sum(len(s.split("\n")) + len(r.split("\n")) for _, s, r, _ in edits) > 80:
            return {"stage": "diff_too_big"}
        rc, bib = None, False
        for bib in (False, True):
            rc = self.compile(sha, src, bib)
            if rc["ok"]:
                break
        if not rc["ok"]:
            return {"stage": "child_timeout" if rc["timed_out"] else "child_compile_fail"}
        broken = dict(files)
        for f, s, r, _ in edits:
            broken[f] = broken[f].replace(s, r, 1)
        rb = self.compile(parent, src, bib, overrides={f: broken[f] for f in broken if broken[f] != files[f]})
        if rb["timed_out"]:
            return {"stage": "parent_timeout"}
        if rb["ok"]:
            return {"stage": "parent_compiles"}
        row = make_row(src, files, broken, edits, rb, {"bib": bib})
        if not row or not row["error_lines"]:
            return {"stage": "parent_no_located_error"}
        m = m1.re.search(r"File `([^']+)' not found|I can't find file `([^']+)'", row["error_lines"][0]["text"])
        if m:
            missing = Path((m.group(1) or m.group(2)).strip())
            if not any((p.name == missing.name) if missing.suffix else (p.stem == missing.stem) for p in src.rglob("*")):
                return {"stage": "missing_asset"}
        single = len(m1.build_files(rc["log"], set(files))) < 2
        return {"stage": "kept", "row": row, "engine": rc["engine"], "files": files, "single": single,
                "ref": {"text": rc["pdf_text"], "pages": rc["pdf_pages"]}}

    def run(self) -> dict:
        t0 = time.monotonic()
        cs = self.commits()
        if not cs:
            self.stages["clone_failed_or_no_tex_commits"] += 1
        kept = 0
        for sha, parent, msg in cs:
            if kept >= MAX_KEPT or time.monotonic() > self.deadline:
                break
            f = self.diff_filter(sha, parent)
            if isinstance(f, str):
                self.stages[f] += 1; continue
            try:
                res = self.pair(sha, parent, msg, f)
            except Exception as e:  # noqa: BLE001
                res = {"stage": "worker_error"}; self.errs.append(f"{sha[:7]} {e!r}"[:200])
            self.stages[res["stage"]] += 1
            if res["stage"] == "kept":
                kept += 1; self.rows.append(res["row"]); self.refs[res["row"]["doc"]] = res["ref"]
        shutil.rmtree(self.d, ignore_errors=True)
        self.stages["kept_single"] += sum(r["single_file_build"] for r in self.rows)
        return {"repo": self.repo, "n_commits": len(cs), "compiles": self.n_compiles, "local": self.n_local, "compile_secs": round(self.compile_secs), "kept": kept,
                "stages": dict(self.stages), "errs": self.errs, "secs": round(time.monotonic() - t0)}


def walk(args) -> None:
    G2.mkdir(parents=True, exist_ok=True)
    done = {d["repo"] for d in read_jsonl(DONE)}
    seen = {(r["repo"], r["sha"]) for p in (common.DATA / "github_multi.jsonl", OUT, OUT1) for r in read_jsonl(p)}
    todo = [r for r in read_jsonl(REPOS) if r["repo"] not in done]
    random.Random(0).shuffle(todo)
    todo.sort(key=lambda r: "ci" not in r["sources"])  # CI repos first (histories tend to compile)
    todo = todo[:args.limit] if args.limit else todo
    deadline = time.monotonic() + args.hours * 3600
    stats = Counter(json.loads(STATS.read_text())) if STATS.exists() else Counter()
    t0, n, total = time.monotonic(), 0, len(todo)
    logf(f"walk: {total} repos to walk ({len(done)} done before, {len(seen)} commits already mined), {args.workers} workers, {args.hours}h budget")

    def run(r):
        if time.monotonic() > deadline:
            return None
        w = Walker(r, seen, deadline)
        return {**w.run(), "rows": w.rows, "refs": w.refs}

    with ThreadPoolExecutor(args.workers) as ex, DONE.open("a") as df, OUT.open("a") as of, OUT1.open("a") as sf, REFS.open("a") as rf:
        futs = {ex.submit(run, r): r for r in todo}
        for fut in as_completed(futs):
            w = fut.result()
            if w is None:
                continue
            n += 1
            for k, v in w["stages"].items():
                stats[k] += v
            stats["repos_walked"] += 1; stats["commits_compiled"] += w["compiles"]; stats["commits_considered"] += w["n_commits"]
            for row in w.pop("rows"):
                (sf if row["single_file_build"] else of).write(json.dumps(row) + "\n")
                logf(f"  KEPT{' (single)' if row['single_file_build'] else ''} {row['repo']}@{row['sha'][:7]} [{row['error_category']}] "
                     f"fix={row['fix_files']} reported={row['reported_file']}")
            for doc, ref in w.pop("refs").items():
                rf.write(json.dumps({"doc": doc, "ref": ref}) + "\n")
            df.write(json.dumps(w) + "\n"); df.flush(); of.flush(); sf.flush(); rf.flush()
            if n % 25 == 0 or n == total:
                rate = n / (time.monotonic() - t0)
                STATS.write_text(json.dumps(dict(stats), indent=1))
                logf(f"{n}/{total} repos, kept multi={stats['kept'] - stats['kept_single']} single={stats['kept_single']} "
                     f"compiles={stats['commits_compiled']} eta {(total - n) / max(rate, 1e-6) / 60:.0f}m  "
                     f"{dict(Counter({k: v for k, v in stats.items() if not k.startswith(('repos_', 'commits_'))}).most_common(8))}")
    STATS.write_text(json.dumps(dict(stats), indent=1))
    logf(f"walk done: {dict(stats)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["repos", "walk"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hours", type=float, default=5)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    {"repos": repos, "walk": walk}[args.cmd](args)


if __name__ == "__main__":
    main()
