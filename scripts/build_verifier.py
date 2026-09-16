"""Verifier data from sample_k --dump candidates: (prompt + proposed fix -> yes/no) where yes = the compiled PDF matches the verified
reference (sim >= .95). Only compiled candidates (the inference selector discards the rest), deduplicated by PDF text per prompt,
at most --per positives and negatives per prompt.  build_verifier.py cands_train_M6c_k8.jsonl --out verifier_train.jsonl"""
import argparse
import json
import random

import common

ASK = "\n\nProposed fix:\n{fix}\n\nThe patched document compiles. Does this fix resolve the error correctly and keep the document's content and intent intact? Answer yes or no."


def verifier_prompt(prompt: str, fix: str) -> str:
    return prompt + ASK.format(fix=fix.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default="verifier_train.jsonl")
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--val", type=int, default=200)
    a = ap.parse_args()
    prompts = {}
    for name in ("texse2_train.jsonl", "texse_pool.jsonl", "multi2_train.jsonl", "texse2_pool.jsonl"):
        for l in (common.DATA / name).read_text().splitlines():
            if l.strip():
                r = json.loads(l); prompts[r["id"]] = r["prompt"]
    rng, rows, npos, nneg, nprompts = random.Random(0), [], 0, 0, 0
    for l in (common.DATA / a.dump).read_text().splitlines():
        rec = json.loads(l)
        if rec["id"] not in prompts:
            continue
        seen, pos, neg = set(), [], []
        for c in rec["cands"]:
            if not c["compiled"] or c.get("sim") is None:
                continue
            key = (c.get("pages"), (c.get("pdf_text") or "")[:2000])
            if key in seen:
                continue
            seen.add(key)
            (pos if c["sim"] >= 0.95 else neg).append(c["text"])
        rng.shuffle(pos); rng.shuffle(neg)
        if not pos and not neg:
            continue
        nprompts += 1
        for t in pos[:a.per]:
            rows.append({"id": rec["id"], "prompt": verifier_prompt(prompts[rec["id"]], t), "target": "yes"}); npos += 1
        for t in neg[:a.per]:
            rows.append({"id": rec["id"], "prompt": verifier_prompt(prompts[rec["id"]], t), "target": "no"}); nneg += 1
    rng.shuffle(rows)
    val, train = rows[:a.val], rows[a.val:]
    (common.DATA / a.out).write_text("".join(json.dumps(r) + "\n" for r in train))
    (common.DATA / a.out.replace(".jsonl", "_val.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in val))
    print(f"{nprompts} prompts -> {len(train)} train / {len(val)} val rows ({npos} yes, {nneg} no)")


if __name__ == "__main__":
    main()
