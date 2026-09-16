"""Combine the real multi-file sets (github_multi + github_multi2 + github_forks + texse_multi): merge farm PDF refs into data/refs.json,
freeze a held-out split by repo / question (data/multi2_split.json, multi2_{heldout,train}.jsonl), write data/multi2_report.md."""
import hashlib
import json
from collections import Counter

import common
import mine_github_multi2 as m2

D = common.DATA
SRC = {"github_multi": D / "github_multi.jsonl", "github_multi2": D / "github_multi2.jsonl", "github_forks": D / "github_forks.jsonl", "texse_multi": D / "texse_multi.jsonl"}
HELDOUT_PCT = 30


def key(r: dict) -> str:
    return r["repo"] if "repo" in r else f"q{r['question_id']}"


def main() -> None:
    refs_path = D / "refs.json"
    refs = json.loads(refs_path.read_text())
    new = {x["doc"]: x["ref"] for p in (m2.REFS, common.PROJ / "corpus" / "github_forks" / "refs.jsonl", common.PROJ / "corpus" / "texse_multi" / "refs.jsonl")
           for x in m2.read_jsonl(p)}
    refs.update(new); refs_path.write_text(json.dumps(refs))
    rows = {s: m2.read_jsonl(p) for s, p in SRC.items()}
    allr = [r for rs in rows.values() for r in rs]
    old_held = set(json.loads((D / "multi_split.json").read_text())["heldout_repos"])  # github_multi's frozen split stays
    split_path = D / "multi2_split.json"
    prev = json.loads(split_path.read_text()) if split_path.exists() else {"heldout": [], "train": []}  # frozen once written
    old_held |= set(prev["heldout"]); old_train = set(prev["train"]) | {r["repo"] for r in rows["github_multi"]} - old_held
    keys = sorted({key(r) for r in allr})
    held = {k for k in keys if (k in old_held if k in old_held | old_train else
                                int(hashlib.sha1(k.encode()).hexdigest(), 16) % 100 < HELDOUT_PCT)}
    split_path.write_text(json.dumps({"heldout": sorted(held), "train": sorted(set(keys) - held)}, indent=1))
    for name, pred in (("heldout", lambda r: key(r) in held), ("train", lambda r: key(r) not in held)):
        (D / f"multi2_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in allr if pred(r)))
    fstats = json.loads((common.PROJ / "corpus" / "github_forks" / "stats.json").read_text())
    fdone = m2.read_jsonl(common.PROJ / "corpus" / "github_forks" / "done.jsonl")
    forks_all = m2.read_jsonl(common.PROJ / "corpus" / "github_forks" / "forks.jsonl")
    tstats = json.loads((common.PROJ / "corpus" / "texse_multi" / "stats.json").read_text())
    hist = lambda rs, f, n=10: "\n".join(f"| {k} | {v} |" for k, v in Counter(f(r) for r in rs).most_common(n))
    stage_tbl = lambda d: "\n".join(f"| {k} | {v} |" for k, v in Counter(d).most_common())
    ex = [r for s in ("github_forks", "texse_multi") for r in sorted(rows[s], key=lambda r: r["reported_file"] in r["fix_files"])[:2]][:3]
    md = ["# Real multi-file compile-fix sets, round 2 (2026-09-08)", "",
          "Project-lane rows (`lane=project`, `file_edits`): parent/broken project fails on the farm (TeX Live 2026) with a located error, "
          "child/fixed compiles; edits round-trip exactly. Refs (`data/refs.json`) come from the farm's child compile.", "",
          "| source | rows | keys (repo/question) | held-out rows |", "|---|---|---|---|"]
    md += [f"| {s} | {len(rs)} | {len({key(r) for r in rs})} | {sum(key(r) in held for r in rs)} |" for s, rs in rows.items()]
    md += [f"| **total** | {len(allr)} | {len(keys)} | {sum(key(r) in held for r in allr)} |", "",
           f"Split frozen in `data/multi2_split.json` (github_multi keeps `multi_split.json`; new keys hashed, {HELDOUT_PCT}% held out) -> "
           f"`data/multi2_heldout.jsonl` / `data/multi2_train.jsonl`.", "",
           "## A. GitHub template fork networks (`scripts/mine_github_forks.py`)", "",
           f"{len(forks_all)} forks ahead of upstream (of {len(set(f['upstream'] for f in forks_all))} templates, pushed forks only), "
           f"{fstats.get('repos_walked', 0)} walked in the time budget, {fstats.get('commits_considered', 0)} own tex-touching commits considered, "
           f"{fstats.get('commits_compiled', 0)} compiled. Upstreams of kept rows: {dict(Counter(r['upstream'] for r in rows['github_forks']))}.", "",
           "| stage (per commit) | n |", "|---|---|",
           stage_tbl({k: v for k, v in fstats.items() if not k.startswith(("repos_", "commits_", "kept_single"))}), "",
           "## B. TeX.SE companion-file questions (`scripts/mine_texse_multi.py`)", "",
           f"Parse: {tstats['parse']}", "", "| stage (per question) | n |", "|---|---|", stage_tbl(tstats.get("mine", {})), "",
           f"Companion kinds in kept rows: {dict(Counter(k.rsplit('.', 1)[-1] for r in rows['texse_multi'] for k in r['companions']))}; "
           f"fix in a companion file: {sum(r['fix_files'] != ['main.tex'] for r in rows['texse_multi'])}; answer used: "
           f"{dict(Counter(r['fix_answer'] for r in rows['texse_multi']))}.", "",
           "## Error histogram (first located error, normalized, all new rows)", "", "| error | n |", "|---|---|",
           hist(rows["github_forks"] + rows["texse_multi"], lambda r: r["error_category"], 12), "",
           "## Licences", "", "| source | licence | n |", "|---|---|---|",
           "\n".join(f"| github_forks | {k} | {v} |" for k, v in Counter(f"{r['license']} (upstream {r['upstream_license']})" for r in rows["github_forks"]).most_common(8)),
           "\n".join(f"| texse_multi | {k} | {v} |" for k, v in Counter(r.get("license") for r in rows["texse_multi"]).most_common()), "",
           "## Examples", ""]
    for r in ex:
        md.append(f"- `{r['src_dir']}` <{r['url']}> ({r['license']}): error `{r['error_lines'][0]['text'][:70]}` reported in `{r['reported_file']}`, "
                  f"fix in {r['fix_files']} ({len(r['file_edits'])} edit(s), {r['n_files']} source files).")
    md += ["", "## What limited yield", "",
           "- Forks: only ~10% of forks have pushes and ~50% of those are ahead; most own commits add chapters + figures in one commit "
           "(`non_text_change`, `too_many_files`, `diff_too_big`) or move between two compiling states (`parent_compiles`). CJK theses need "
           "xelatex (~100 s per compile on the 1-CPU farm) and some forks pin Windows fonts (`SimSun`) -> `child_compile_fail`; "
           "thuthesis forks ship only `.dtx`, the class is docstripped locally before compiling.",
           "- TeX.SE: most questions with a companion block reference nothing external (the block is a log or snippet); of the rest, the "
           "companion is often an excerpt (`child_compile_fail`), the asker's project compiles under TL2026 (`parent_compiles`), or the answer "
           "is prose. `.bib` companions are only usable when the error surfaces in the TeX log (bibtex errors carry no file/line).",
           f"- Fork rows are unlicensed user content on top of {'/'.join(sorted({str(f['upstream_license']) for f in forks_all}))} templates: train/eval only, no redistribution."]
    (D / "multi2_report.md").write_text("\n".join(md) + "\n")
    print(f"refs +{len(new)}; rows {dict((s, len(rs)) for s, rs in rows.items())}; heldout {sum(key(r) in held for r in allr)}/{len(allr)}; wrote data/multi2_report.md ({len(md)} lines)")


if __name__ == "__main__":
    main()
