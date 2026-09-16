"""Add the companion-file context block (common.context_block) to every project-lane prompt; keep the old prompt as
prompt_v1 and recompute single_shot_visible (all fix sites present in the prompt).  usage: rebuild_prompts.py [sets...]"""
import json
import os
import sys

import common

SETS = ["multi2_train", "multi2_heldout", "skew_train", "skew_multi_heldout", "project_train_all", "project_eval",
        "multi_synth", "github_multi"]


def visible(row: dict, prompt: str) -> bool:
    return all(any(l.strip()[:60] in prompt for l in (e["search"].splitlines() or [e["search"]]) if len(l.strip()) >= 8)
               for e in row["file_edits"]) if row.get("file_edits") else True


def rebuild(row: dict) -> dict:
    base = row.get("prompt_v1") or row["prompt"]
    ctx = common.context_block(row["src_dir"], row["reported_file"], row["error_lines"], row.get("log_tail", ""))
    cut = next((base.find(k) for k in ("\n\nCurrent content of `", "\n\nCompile log (tail):") if k in base), len(base))
    prompt = base[:cut] + ctx + base[cut:]
    return {**row, "prompt_v1": base, "prompt": prompt, "single_shot_visible": visible(row, prompt), "ctx_chars": len(ctx)}


for name in sys.argv[1:] or SETS:
    path = common.DATA / f"{name}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out, v0, v1, lens = [], 0, 0, []
    for r in rows:
        if r.get("src_dir") and (r.get("lane") == "project" and (common.PROJ / r["src_dir"]).exists() or os.path.isdir(r["src_dir"])):
            v0 += bool(r.get("single_shot_visible")); r = rebuild(r); v1 += r["single_shot_visible"]; lens.append(len(r["prompt"]))
        out.append(r)
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    lens.sort()
    print(f"{name:18s} rows={len(rows):5d} rebuilt={len(lens):5d} visible {v0}->{v1}  prompt chars p50={lens[len(lens)//2] if lens else 0} "
          f"p90={lens[int(len(lens)*.9)] if lens else 0} max={lens[-1] if lens else 0}  with_ctx={sum(r.get('ctx_chars',0)>0 for r in out)}")
