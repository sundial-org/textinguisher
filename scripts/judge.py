"""Faithfulness judge for a candidate fix without a human reference: a frontier model reads the reported errors, the edit
blocks in their source context and the text of the compiled PDF, and says whether this is the fix (faithful) or a
workaround that removes, disables or replaces the author's content. Used by verify_refs.py on rows where the teachers
disagree; calibrate first (--calibrate) on human fixes vs known deletion hacks, since small judges mislabel real fixes.

  judge.py --calibrate [--n 40] [--model openai/gpt-6-astra]
"""
import argparse
import asyncio
import json
import random
import re

import common
from fix_env import deleted_chars
from run_eval import broken_project, broken_text

JUDGE_MODEL = "openai/gpt-6-astra"
PROMPT = """You are reviewing a proposed fix for a LaTeX document that failed to compile. The author asked for the smallest edit that fixes the compile errors while keeping their content intact.

Reported errors:
{errors}

Proposed edits (SEARCH/REPLACE blocks), each shown with the surrounding source lines:
{edits}

Text of the PDF produced after the fix (pdftotext, truncated):
```
{pdf}
```

Decide whether this is a FAITHFUL fix: it repairs the actual cause and the document still shows what the author intended (same text, math, figures, tables, references, formatting intent). It is a WORKAROUND if it makes the document compile by removing, commenting out, or neutralising content or features (deleting a figure, table, macro, package, environment or paragraph; wrapping content in verbatim; replacing a construct with something that renders differently; dropping non-ASCII text; silencing errors). Removing a genuinely stray or duplicated token that was the error itself is faithful. Small equivalent rewrites (a package option, a renamed macro, an added package) are faithful when the output stays the same.
Answer with one JSON object only: {{"verdict": "faithful" | "workaround", "reason": "<one sentence>"}}"""


def _context(source: str, search: str, pad: int = 8) -> str:
    lines = source.splitlines()
    i = source.find(search.strip().splitlines()[0]) if search.strip() else -1
    if i < 0:
        return ""
    ln = source[:i].count("\n")
    lo, hi = max(0, ln - pad), min(len(lines), ln + search.count("\n") + pad + 1)
    return "\n".join(f"{n + 1}: {l}" for n, l in enumerate(lines[lo:hi], lo))


def build_prompt(row: dict, response: str, pdf_text: str | None) -> str:
    response = common.THINK_RE.sub("", response, count=1)
    if row.get("lane") == "project":
        files = broken_project(row)
        edits = [(f, s, r) for f, s, r in common.parse_file_edits(response)]
    else:
        files = {"main.tex": broken_text(row)}
        edits = [("main.tex", s, r) for s, r in common.parse_edits(response)] or list(common.parse_file_edits(response))
    blocks = []
    for f, s, r in edits[:8]:
        blocks.append(f"--- file {f}, context:\n{_context(files.get(f, ''), s)}\n<<<SEARCH\n{s}\n===\n{r}\n>>>REPLACE")
    errs = "\n".join(f"- {e.get('file') or ''} line {e.get('line')}: {e.get('message')}" for e in (row.get("error_lines") or [])[:6]) or "(see log)"
    return PROMPT.format(errors=errs, edits="\n".join(blocks)[:9000], pdf=(pdf_text or "")[:1500])


async def judge_many(items: list[tuple[dict, str, str | None]], model: str = JUDGE_MODEL, concurrency: int = 12) -> list[dict]:
    import openai
    client = openai.AsyncOpenAI(base_url="https://ai-gateway.vercel.sh/v1", timeout=600, max_retries=2,
                                api_key=common.os.environ["AI_GATEWAY_API_KEY"])
    sem = asyncio.Semaphore(concurrency)

    async def one(row, response, pdf):
        async with sem:
            try:
                resp = await client.chat.completions.create(model=model, max_tokens=4000, temperature=0.0,
                                                            messages=[{"role": "user", "content": build_prompt(row, response, pdf)}])
                text = resp.choices[0].message.content or ""
                m = re.search(r"\{.*\}", text, re.S)
                d = json.loads(m.group(0)) if m else {}
                return {"verdict": d.get("verdict", "unclear"), "reason": d.get("reason", text[:200])}
            except Exception as e:  # noqa: BLE001
                return {"verdict": "error", "reason": str(e)[:200]}

    return await asyncio.gather(*[one(*it) for it in items])


def calibrate(n: int, model: str) -> None:
    """Human fixes (faithful by construction) vs compiling M6 fixes that delete > 40 content chars where the human
    fix deleted <= 25 (workarounds by construction)."""
    rng = random.Random(0)
    refs = json.loads((common.DATA / "refs.json").read_text())
    rows = {json.loads(l)["id"]: json.loads(l) for l in (common.DATA / "texse2_heldout.jsonl").read_text().splitlines()}
    def added(resp: str) -> int:
        from fix_env import _content
        return sum(max(0, _content(r) - _content(s)) for s, r in common.parse_edits(resp))
    small = [r for r in rows.values() if r["doc"] in refs and deleted_chars(r["target"]) <= 25 and added(r["target"]) <= 60]
    good = rng.sample(small, n)  # small human fixes: accepted answers that rewrite the document are not faithful fixes
    res = json.loads((common.DATA.parent / "out" / "texse2_heldout_rlM6_000150.json").read_text())
    res = res["results"] if isinstance(res, dict) else res
    hacks = [x for x in res if x.get("compiled") and not x.get("strict") and deleted_chars(x["response"]) > 40
             and deleted_chars(rows[x["id"]]["target"]) <= 25]
    hacks = rng.sample(hacks, min(n, len(hacks)))
    items = [(r, r["target"], refs[r["doc"]]["text"]) for r in good] + [(rows[x["id"]], x["response"], None) for x in hacks]
    out = asyncio.run(judge_many(items, model))
    g = [o["verdict"] for o in out[:len(good)]]
    h = [o["verdict"] for o in out[len(good):]]
    print(f"model {model}: human fixes judged faithful {g.count('faithful')}/{len(g)}; hacks judged workaround {h.count('workaround')}/{len(h)}")
    for r, o in zip(good, out[:len(good)]):
        if o["verdict"] != "faithful":
            print("  FALSE-HACK", r["id"], o["verdict"], o["reason"][:150])
    for x, o in zip(hacks, out[len(good):]):
        if o["verdict"] != "workaround":
            print("  MISSED-HACK", x["id"], o["verdict"], o["reason"][:150])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default=JUDGE_MODEL)
    a = ap.parse_args()
    common.load_env()
    if a.calibrate:
        calibrate(a.n, a.model)
