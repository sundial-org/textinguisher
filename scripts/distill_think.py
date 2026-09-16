"""Rationalised diagnoses for the thinking-SFT stage: a teacher sees the broken doc, the errors AND the known-good
(accepted-answer) fix, and writes the short diagnosis that leads to exactly that fix. Output rows carry both a
<think>-prefixed and a plain target, so SFT learns both modes and RL (LENGTH_W) decides when reasoning pays.

  distill_think.py --data texse_train.jsonl,texse2_train.jsonl --out think_train.jsonl
"""
import argparse
import asyncio
import json

import common

ASK = ("\n\nThe correct fix (already known) is:\n{target}\n\nWrite the diagnosis a LaTeX expert would give before making "
       "exactly this fix: what the error message means here, the root cause in the source, and why this edit fixes it. "
       "At most 80 words, plain text, no code fences, no restating the edit.")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma list of data/<file> with prompt+target")
    ap.add_argument("--teacher", default="google/gemini-3.7-flash")
    ap.add_argument("--out", default="think_train.jsonl")
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    common.load_env()
    import openai
    client = openai.AsyncOpenAI(base_url="https://ai-gateway.vercel.sh/v1", api_key=common.os.environ["AI_GATEWAY_API_KEY"])
    out = common.DATA / args.out
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    rows = [r for f in args.data.split(",") for r in map(json.loads, (common.DATA / f).read_text().splitlines()) if r["id"] not in done]
    sem, n = asyncio.Semaphore(args.concurrency), 0
    print(f"{len(rows)} rows to do ({len(done)} done) -> {out}", flush=True)

    async def one(r):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=args.teacher, max_tokens=300, temperature=0.0,
                    messages=[{"role": "user", "content": r["prompt"] + ASK.format(target=r["target"])}])
                return r, (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                return r, ""

    with out.open("a") as fh:
        for coro in asyncio.as_completed([one(r) for r in rows]):
            r, think = await coro
            if think:
                fh.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "target": r["target"], "think": think}) + "\n")
                fh.flush()
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(rows)}", flush=True)
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
