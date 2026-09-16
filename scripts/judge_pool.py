"""Listwise selection by a judge model: for each prompt whose pool (passk --dump --dump-text) has >= 2 distinct compiling PDFs, show the
error prompt and one representative candidate per PDF group (fewest deleted chars) and ask which candidate fixes the error while keeping
the content. Reports compiles / kept / exact of the judged selection vs kept-first consensus.  judge_pool.py out/cands_<tag>.jsonl [model]"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import common

dump, model = sys.argv[1], (sys.argv[2:] or [common.MODEL])[0]
common.load_env()
import tinker
from tinker import types
from tml_renderers import chat

rows = {json.loads(l)["id"]: json.loads(l) for l in (common.DATA / "texse2_heldout.jsonl").read_text().splitlines() if l.strip()}
pools = [json.loads(l) for l in open(dump)]
sampler = tinker.ServiceClient().create_sampling_client(model_path=model) if model.startswith("tinker://") else tinker.ServiceClient().create_sampling_client(base_model=model)
renderer = common.get_renderer()
ASK = ("\n\nSeveral candidate edits were tried and ALL of them make the document compile, but only some are correct fixes; others hide the "
       "error by deleting or rewriting content, or change the document's meaning. Pick the candidate that fixes the actual cause and keeps the "
       "content intact.\n\n{cands}\n\nAnswer with the candidate number only.")


def groups(pool):
    g = {}
    for c in pool["cands"]:
        if c["compiled"] and c.get("pdf_key") is not None:
            k = tuple(c["pdf_key"]) if isinstance(c["pdf_key"], list) else c["pdf_key"]
            g.setdefault(k, []).append(c)
    return [min(v, key=lambda c: (c["deleted"], c["len"])) | {"votes": len(v)} for v in g.values()]


def ask(pool, reps):
    body = "\n\n".join(f"Candidate {i + 1}:\n{c['text'].strip()}" for i, c in enumerate(reps))
    msgs = [chat.Message(content=chat.Text(common.system_prompt()), author=chat.Author(chat.AuthorKind.System)),
            chat.Message(content=chat.Text(rows[pool["id"]]["prompt"] + ASK.format(cands=body)), author=chat.Author(chat.AuthorKind.User))]
    spans, parser = renderer.render_for_completion_with_effort(msgs, 0.9)
    from tml_renderers.tinker import token_spans_to_tinker_model_input
    fut = sampler.sample(prompt=token_spans_to_tinker_model_input(spans), sampling_params=types.SamplingParams(max_tokens=8000, temperature=0.0, stop=renderer.stop()), num_samples=1)
    return fut, parser


def parse(fut, parser, n):
    try:
        seq = fut.result(timeout=900).sequences[0]
        text = "\n".join(m.content.text for m in parser.parse_tokens(seq.tokens) if isinstance(m.content, chat.Text))
    except Exception:  # noqa: BLE001
        return None
    t = text.strip()
    m = re.findall(r"[Cc]andidate\s*(\d+)", t[-300:]) or re.findall(r"\d+", t[-40:]) or re.findall(r"\d+", t)
    return int(m[-1]) - 1 if m and 1 <= int(m[-1]) <= n else None


n = len(pools); stats = {"consensus": [0, 0, 0], "judge": [0, 0, 0]}; asked = agreed = fell = 0
todo = []
for pool in pools:
    reps = groups(pool)
    comp = [c for c in pool["cands"] if c["compiled"]]
    cons = min(comp, key=lambda c: (c["deleted"] > 40, -c["votes"], c["deleted"])) if comp else None
    if cons:
        stats["consensus"][0] += 1; stats["consensus"][1] += cons["deleted"] <= 40; stats["consensus"][2] += cons["strict"]
    if len(reps) >= 2:
        todo.append((pool, reps, cons)); asked += 1
    elif cons:
        stats["judge"][0] += 1; stats["judge"][1] += cons["deleted"] <= 40; stats["judge"][2] += cons["strict"]
print(f"{n} prompts, {asked} with >= 2 distinct compiling PDFs -> judged by {model}", flush=True)
with ThreadPoolExecutor(16) as ex:
    futs = [(pool, reps, cons, *ask(pool, reps)) for pool, reps, cons in todo]
    for pool, reps, cons, fut, parser in futs:
        j = parse(fut, parser, len(reps))
        pick = reps[j] if j is not None else cons
        fell += j is None; agreed += j is not None and pick["strict"] == cons["strict"] and pick["deleted"] == cons["deleted"]
        stats["judge"][0] += 1; stats["judge"][1] += pick["deleted"] <= 40; stats["judge"][2] += pick["strict"]
for k, (c, kp, e) in stats.items():
    print(f"{k:10s} compiles {100 * c / n:.1f} kept {100 * kp / n:.1f} exact {100 * e / n:.1f}")
print(f"judge answered {asked - fell}/{asked}; same pick as consensus on {agreed}; oracle exact {100 * sum(any(c['compiled'] and c['strict'] for c in p['cands']) for p in pools) / n:.1f}")
