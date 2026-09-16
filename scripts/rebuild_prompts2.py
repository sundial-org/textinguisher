"""Rebuild single-file prompts with the fixed log parser (named undefined macros, package-internal line numbers flagged):
recompile each row's broken document on the farm, re-parse, and write data/<set>_p2.jsonl (prompt + error_lines updated,
everything else kept).  rebuild_prompts2.py texse2_heldout [texse2_train ...]"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common
from mine_texse2 import V2_FULL_LINES, V2_MAX_CHARS
from run_eval import broken_text


def one(row: dict) -> dict:
    broken = broken_text(row)
    rb = common.compile_project(Path(row["src_dir"]), [], overrides={"main.tex": broken}, timeout=120, backend="modal")
    files = {p.name for p in Path(row["src_dir"]).iterdir()}
    err_lines = common.error_lines_from_items(rb["items"], "main.tex", files)
    prompt = common.build_fix_prompt("main.tex", err_lines, rb["log"], broken, full_lines=V2_FULL_LINES, max_chars=V2_MAX_CHARS)
    return {**row, "prompt_v1": row.get("prompt_v1") or row["prompt"], "prompt": prompt, "error_lines": err_lines,
            "log_tail": rb["log"][-common.MAX_LOG_CHARS:], "rebuilt": bool(rb["error_items"])}


if __name__ == "__main__":
    common.load_env()
    for name in sys.argv[1:]:
        rows = [json.loads(l) for l in (common.DATA / f"{name}.jsonl").read_text().splitlines() if l.strip()]
        with ThreadPoolExecutor(48) as ex:
            out = list(ex.map(one, rows))
        changed = sum(o["prompt"] != o["prompt_v1"] for o in out)
        (common.DATA / f"{name}_p2.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
        print(f"{name}_p2.jsonl: {len(out)} rows, {changed} prompts changed, {sum(not o['rebuilt'] for o in out)} rows compiled clean (kept old prompt structure) done", flush=True)
