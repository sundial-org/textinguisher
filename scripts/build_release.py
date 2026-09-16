"""Build release/ (public benchmark package) from data/ + corpus/, deterministically.
Usage: build_release.py [--check]   (--check: only verify an existing release/)"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import common

REL = common.PROJ / "release"
TEXSE = common.PROJ / "corpus" / "texse"
SINGLE = ["texse_heldout", "texse2_heldout", "texse_train", "texse2_train", "texse_pool", "texse2_pool"]
MULTI = ["multi2_heldout", "multi2_train"]
SKEW = ["skew_eval", "skew_multi_heldout", "skew_train"]
DROP = {"src_dir", "bundles", "seed", "compile_seconds", "ctx_chars", "query", "stars", "single_file_build",
        "upstream", "upstream_license", "ahead_by", "commit_message"}
SKEW_KEEP = ["id", "doc", "arxiv_id", "year", "hist_year", "signature", "first_error", "missing_file", "n_files", "engine",
             "error_lines", "license"]
OSI_CC = {"MIT", "Apache-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-3.0", "BSD-2-Clause", "BSD-3-Clause", "LPPL-1.3c",
          "CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0"}
BUILD_EXT = {".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".blg"}
ABS_RE = re.compile(r"/(Users|home|private|tmp|var/folders)/")
LOCAL_RE = re.compile(r"/Users/" + r"(\\n|\n)?".join(Path.home().name) + "|latex-fix/")  # this machine's paths, even wrapped
SE = "https://tex.stackexchange.com"


def rows(name):
    return [json.loads(l) for l in open(common.DATA / f"{name}.jsonl")]


def xml_attrs(line):
    return dict(re.findall(r' (\w+)="([^"]*)"', line))


def texse_meta(post_ids):
    """post id -> {OwnerUserId, OwnerDisplayName, ContentLicense, CreationDate} from Posts.xml, plus user id -> name."""
    if not (TEXSE / "Users.xml").exists() and (TEXSE / "tex.stackexchange.com.7z").exists():
        subprocess.run(["7zz", "e", "-y", str(TEXSE / "tex.stackexchange.com.7z"), "Users.xml", f"-o{TEXSE}"], check=True,
                       capture_output=True)
    posts, users = {}, {}
    with open(TEXSE / "Posts.xml", encoding="utf-8") as f:
        for line in f:
            m = re.match(r'\s*<row Id="(\d+)"', line)
            if m and m.group(1) in post_ids:
                posts[m.group(1)] = xml_attrs(line)
    uids = {a["OwnerUserId"] for a in posts.values() if "OwnerUserId" in a}
    if (TEXSE / "Users.xml").exists():
        with open(TEXSE / "Users.xml", encoding="utf-8") as f:
            for line in f:
                m = re.match(r'\s*<row Id="(-?\d+)"', line)
                if m and m.group(1) in uids:
                    users[m.group(1)] = xml_attrs(line).get("DisplayName")
    return posts, users


def license_for(created):
    return "CC-BY-SA-2.5" if created < "2011-04-08" else "CC-BY-SA-3.0" if created < "2018-05-02" else "CC-BY-SA-4.0"


def attribute(r, posts, users):
    """TeX.SE row: licence from the dump's ContentLicense (date rule as fallback), URLs, author names + profile links."""
    q, a = str(r["question_id"]), str(r.get("answer_id") or r.get("accepted_answer_id") or "")
    r["question_url"] = f"{SE}/q/{q}"
    if a:
        r["answer_url"] = f"{SE}/a/{a}"
    for key, pid in (("question", q), ("answer", a)):
        p = posts.get(pid)
        if p is None:
            continue
        lic = p.get("ContentLicense", "").replace(" ", "-") or license_for(p["CreationDate"][:10])
        r["license" if key == "question" else "answer_license"] = lic
        uid = p.get("OwnerUserId")
        r[f"{key}_author"] = {"user_id": int(uid) if uid else None,
                              "name": users.get(uid) or p.get("OwnerDisplayName"),
                              "url": f"{SE}/users/{uid}" if uid else None}
    r.setdefault("license", license_for(str(r.get("created", "9999"))))


def scrub(r, src_dir):
    """Drop internals; rewrite the local project path to projects/<id> and any other /Users/<name>/ prefix to ~/."""
    s = json.dumps({k: v for k, v in r.items() if k not in DROP}, ensure_ascii=False)
    s = s.replace(src_dir, f"projects/{r['id']}")
    s = re.sub(r"/Users/(?:[^/\s\"'<>()\\]|\\n)+/(Library/)?", "~/", s)  # log lines wrap mid-path: allow a JSON \n
    return json.loads(s)


def copy_project(src_dir, rid):
    dst = REL / "projects" / rid
    for p in sorted(Path(src_dir).rglob("*")):
        if not p.is_file() or p.suffix in BUILD_EXT or p.name.endswith(".synctex.gz") or (p.suffix == ".pdf" and p.with_suffix(".tex").exists()):
            continue
        out = dst / p.relative_to(src_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def write_jsonl(name, out):
    with open(REL / f"{name}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(out)} rows")


def build():
    shutil.rmtree(REL / "projects", ignore_errors=True)
    for p in list(REL.glob("*.jsonl")) + [REL / "refs.json", REL / "refs_verified.json", REL / "MANIFEST.json"]:
        p.unlink(missing_ok=True)
    single, multi = {n: rows(n) for n in SINGLE}, {n: rows(n) for n in MULTI}
    audited = set(json.load(open(common.DATA / "texse2_heldout_drop.json")))  # reference broken / empty output / hidden fix
    single["texse2_heldout"] = [r for r in single["texse2_heldout"] if r["id"] not in audited]
    pids = set()
    for r in [r for rs in single.values() for r in rs] + [r for rs in multi.values() for r in rs if r["source"] == "texse"]:
        pids |= {str(r["question_id"]), str(r.get("answer_id") or r.get("accepted_answer_id") or "")}
    posts, users = texse_meta(pids)
    print(f"TeX.SE posts found {len(posts)}/{len(pids - {''})}, users {len(users)}")
    licences, docs, drop = Counter(), {}, Counter()
    for name, rs in list(single.items()) + list(multi.items()):
        out = []
        for r in rs:
            if r.get("source") == "github":
                if r["doc"].startswith("github_forks") or r.get("license") not in OSI_CC:
                    drop[f"{name}:{r['doc'].split('/')[0]}:{r.get('license')}"] += 1
                    continue
                r["repo_url"] = f"https://github.com/{r['repo']}"
            else:
                attribute(r, posts, users)
            licences[f"{name}:{r['license']}"] += 1
            docs[r["doc"]] = "pool" in name
            copy_project(r["src_dir"], r["id"])
            out.append(scrub(r, r["src_dir"]))
        write_jsonl(name, out)
    for name in SKEW:
        out = []
        for r in rows(name):
            if "license" not in r:
                r["license"] = json.loads((Path(r["src_dir"]).parent / "meta.json").read_text()).get("license")
            licences[f"{name}:{r['license']}"] += 1
            out.append({k: r.get(k) for k in SKEW_KEEP})
        write_jsonl(name, out)
    refs = json.load(open(common.DATA / "refs.json"))
    pool = json.load(open(common.DATA / "refs_pool.json"))
    out = {d: refs[d] for d, is_pool in docs.items() if not is_pool and d in refs}
    out |= {d: {**pool[d], "pseudo": True} for d, is_pool in docs.items() if is_pool and d in pool}
    json.dump(out, open(REL / "refs.json", "w"))
    print(f"refs.json: {len(out)} docs ({sum(v.get('pseudo', False) for v in out.values())} pseudo)")
    # verified references: human fixes whose PDF an independent teacher model or a judge confirmed (build_refs2 / verify_refs, TeX Live 2026)
    ver = json.load(open(common.DATA / "refs2_heldout.json"))
    out = {d: {k: v[k] for k in ("text", "pages", "teacher", "how", "response", "tl") if k in v} for d, v in ver.items() if d in docs}
    json.dump(out, open(REL / "refs_verified.json", "w"))
    print(f"refs_verified.json: {len(out)} docs")
    print("dropped:", dict(drop))
    print("licences:", json.dumps(licences, indent=1))
    manifest = []
    for p in sorted(REL.glob("*")):
        if p.is_file():
            n = sum(1 for _ in open(p)) if p.suffix == ".jsonl" else None
            manifest.append({"file": p.name, "rows": n, "bytes": p.stat().st_size,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    proj = [p for p in (REL / "projects").rglob("*") if p.is_file()]
    manifest.append({"file": "projects/", "files": len(proj), "bytes": sum(p.stat().st_size for p in proj)})
    json.dump(manifest, open(REL / "MANIFEST.json", "w"), indent=1)


def check():
    hits, total = Counter(), 0
    for p in REL.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            t = p.read_text(errors="ignore")
            assert not LOCAL_RE.search(t), p
            hits["projects/" if p.parent != REL else p.name] += len(ABS_RE.findall(t))
    key = lambda r: {r["id"], str(r.get("question_id", "")), str(r.get("arxiv_id", "")), r.get("repo", "")} - {""}
    lines = lambda r: (REL / "projects" / r["id"] / "main.tex").exists() and frozenset(
        l.strip() for l in (REL / "projects" / r["id"] / "main.tex").read_text(errors="ignore").splitlines() if l.strip())
    held, train, hdocs, tdocs = set(), set(), {}, {}
    for p in REL.glob("*.jsonl"):
        for r in map(json.loads, open(p)):
            is_held = "heldout" in p.name or "eval" in p.name
            (held if is_held else train).update(key(r))
            if s := lines(r):
                (hdocs if is_held else tdocs)[r["id"]] = s
    assert not held & train, sorted(held & train)[:10]
    by_size = defaultdict(list)
    for k, s in tdocs.items():
        by_size[len(s)].append((k, s))
    near = [(hk, tk) for hk, hs in hdocs.items() for m in range(int(len(hs) * 0.9), int(len(hs) / 0.9) + 1)
            for tk, ts in by_size[m] if len(hs & ts) / (len(hs) + m - len(hs & ts)) >= 0.9]
    for m in json.load(open(REL / "MANIFEST.json")):
        print(f"{m['file']:24} {m.get('rows') or m.get('files') or '':>6} {m['bytes'] / 1e6:8.1f} MB")
    print(f"total {total / 1e6:.1f} MB; local paths: 0; held-out/train overlap: 0")
    print(f"near-duplicates (Jaccard >= 0.9 on normalized main.tex lines, held-out vs train/pool): {near}")
    print("absolute paths inside third-party content (TeX.SE posts / repo files):", dict(hits))


if __name__ == "__main__":
    if "--check" not in sys.argv:
        build()
    check()
