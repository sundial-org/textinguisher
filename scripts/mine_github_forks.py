"""Thesis/template fork networks: forks ahead of upstream -> walk only their own commits with the
mine_github_multi2 Walker (parent fails / child compiles), rows -> data/github_forks.jsonl.

  mine_github_forks.py forks [--templates a/b,c/d] [--max-forks 1500]   # -> corpus/github_forks/forks.jsonl
  COMPILE_BACKEND=modal MODAL_PROFILE=sundial-prod LOCAL_SLOTS=0 mine_github_forks.py walk [--limit 5] [--hours 4] [--workers 96]
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import common
import mine_github_multi2 as m2

FORKS = common.PROJ / "corpus" / "github_forks"
# re-point the multi2 walker (paths are read at call time) instead of copying it
m2.G2, m2.DEPTH, m2.TIMEOUT = FORKS, 100, 280
m2.REPOS, m2.DONE, m2.STATS, m2.REFS = FORKS / "forks.jsonl", FORKS / "done.jsonl", FORKS / "stats.json", FORKS / "refs.jsonl"
m2.OUT, m2.OUT1, m2.LOG = common.DATA / "github_forks.jsonl", FORKS / "single.jsonl", common.PROJ / "out" / "mine_github_forks.log"
TEMPLATES = ["tuna/thuthesis", "mohuangrui/ucasthesis", "sjtug/SJTUThesis", "TheNetAdmin/zjuthesis", "ustctug/ustcthesis",
             "kks32/phd-thesis-template", "hithesis/hithesis", "COPCSE-NTNU/thesis-NTNU", "bdebye/thesisuestc", "fwalch/tum-thesis-latex",
             "andygrunwald/FOM-LaTeX-Template", "stone-zeng/fduthesis", "whutug/whu-thesis", "fmarotta/kaobook", "ElegantLaTeX/ElegantBook",
             "ElegantLaTeX/ElegantPaper", "ElegantLaTeX/ElegantNote", "kourgeorge/arxiv-style", "Tufte-LaTeX/tufte-latex", "posquit0/Awesome-CV"]
CJK_RE = re.compile(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{(?:[^}]*/)?(?:thuthesis|ucasthesis|sjtuthesis|zjuthesis|ustcthesis|thesisuestc|whu-thesis|"
                    r"fduthesis|hithesis|ctex\w*|awesome-cv)\}|\\documentclass\s*\[[^\]]*(?:lang=cn|\bcn\b|chinese)[^\]]*\]|\\usepackage\s*(?:\[[^\]]*\])?\s*\{[^}]*\b(?:ctex|xeCJK)\b")
m2.m1.SHELL_RE = re.compile(r"(?m)^[^%\n]*(?:\\usepackage(?:\[[^\]]*\])?\{[^}]*\b(?:minted|pythontex|gnuplottex|svg)\b|-shell-escape)")  # thuthesis.cls has a \write18 for the spine only
_pick = common.pick_engine
common.pick_engine = lambda t: "xelatex" if _pick(t) == "pdflatex" and CJK_RE.search(t) else _pick(t)  # CJK theses / fontspec classes refuse pdflatex
log = m2.logf


def forks(args) -> None:
    FORKS.mkdir(parents=True, exist_ok=True)
    have = {f["repo"] for f in m2.read_jsonl(m2.REPOS)}
    for up in (args.templates.split(",") if args.templates else TEMPLATES):
        meta = json.loads(m2.sh(["gh", "api", f"repos/{up}", "--jq", "{b: .default_branch, l: .license.spdx_id}"]).stdout)
        cand = []
        for page in range(1, 60):
            p = m2.sh(["gh", "api", f"repos/{up}/forks?per_page=100&sort=newest&page={page}"])
            if p.returncode:
                log(f"{up} forks p{page}: {p.stderr.decode()[:80]}"); time.sleep(30); continue
            items = json.loads(p.stdout)
            cand += [f for f in items if f["pushed_at"] > f["created_at"] and f["full_name"] not in have]
            if len(items) < 100 or len(cand) >= args.max_forks:
                break

        def compare(f):
            p = m2.sh(["gh", "api", f"repos/{up}/compare/{meta['b']}...{f['owner']['login']}:{f['default_branch']}?per_page=250",
                       "--jq", "{ahead_by, shas: [.commits[].sha]}"])
            if p.returncode:
                return None
            c = json.loads(p.stdout)
            return {"repo": f["full_name"], "upstream": up, "upstream_license": meta["l"], "license": (f["license"] or {}).get("spdx_id"),
                    "fork": True, "size": f["size"], "stars": f["stargazers_count"], "default_branch": f["default_branch"],
                    "ahead_by": c["ahead_by"], "ahead_shas": c["shas"], "sources": ["forks"]}

        n = 0
        with ThreadPoolExecutor(4) as ex, m2.REPOS.open("a") as fh:
            for fut in as_completed([ex.submit(compare, f) for f in cand[:args.max_forks]]):
                r = fut.result()
                if r and 1 <= r["ahead_by"] <= 250:
                    fh.write(json.dumps(r) + "\n"); fh.flush(); n += 1
        log(f"{up}: {len(cand)} pushed forks -> {n} ahead of upstream")


class ForkWalker(m2.Walker):
    def commits(self):
        ahead = set(self.r["ahead_shas"])
        return [c for c in super().commits() if c[0] in ahead]

    def check(self, sha, parent, src, *a):
        for ins in src.glob("*.ins"):  # thuthesis ships .dtx/.ins only: docstrip the class locally, else TeX Live's newer one is picked up
            if not (src / (ins.stem + ".cls")).exists():
                m2.sh(["xetex", "-interaction=batchmode", ins.name], cwd=src, timeout=120)
                (src / (ins.stem + ".log")).unlink(missing_ok=True)
        return super().check(sha, parent, src, *a)

    def pair(self, sha, parent, msg, changed):
        res = super().pair(sha, parent, msg, changed)
        if res["stage"] == "kept":
            slug = res["row"]["doc"].split("/", 1)[1]
            res["row"].update(id=f"github_forks_{slug}", doc=f"github_forks/{slug}", query="forks", upstream=self.r["upstream"],
                              upstream_license=self.r["upstream_license"], ahead_by=self.r["ahead_by"])
        return res


m2.Walker = ForkWalker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["forks", "walk"])
    ap.add_argument("--templates", default="")
    ap.add_argument("--max-forks", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hours", type=float, default=4)
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()
    {"forks": forks, "walk": m2.walk}[args.cmd](args)


if __name__ == "__main__":
    main()
