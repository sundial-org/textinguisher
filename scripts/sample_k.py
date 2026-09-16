"""Expert iteration, sampling stage: K samples per training prompt from our own checkpoint at T=1, each compiled and scored
against the verified reference; the exact ones (sim >= KEEP_SIM) become SFT targets (data/<out>: prompt, target, id, src).
On-policy data, so the follow-up SFT does not shift the model the way teacher SFT did.

  sample_k.py --ckpt tinker://.../sampler_weights/000120 --n 2000 --k 8 --out ei_round1.jsonl
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import common
from run_eval import score_response

KEEP_SIM = 0.95


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="ei_round1.jsonl")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--dump", default=None, help="also write every candidate with compiled/sim/deleted to data/<dump> (verifier training data)")
    ap.add_argument("--sets", default="texse2_train.jsonl,texse_pool.jsonl,multi2_train.jsonl")
    args = ap.parse_args()
    common.load_env()
    import tinker
    from tinker import types
    from tml_renderers import chat
    from tml_renderers.tinker import token_spans_to_tinker_model_input

    refs = common.load_refs()
    val = {json.loads(l)["id"] for l in (common.DATA / "rl_val2.jsonl").read_text().splitlines()}
    rows = []
    for name in args.sets.split(","):
        rows += [json.loads(l) for l in (common.DATA / name).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["doc"] in refs and r["id"] not in val and len(r["prompt"]) <= 20000]
    random.Random(21).shuffle(rows)
    rows = rows[: args.n]
    out = common.DATA / args.out
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} prompts with a verified reference, {len(todo)} to sample x{args.k} from {args.ckpt}", flush=True)
    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(model_path=args.ckpt)
    renderer = common.get_renderer()
    t0, kept, seen = time.monotonic(), 0, 0
    dump = (common.DATA / args.dump).open("a") if args.dump else None
    with out.open("a") as fh, ThreadPoolExecutor(64) as ex:
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            futs = []
            for row in chunk:
                msgs = [chat.Message(content=chat.Text(common.system_prompt()), author=chat.Author(chat.AuthorKind.System)),
                        chat.Message(content=chat.Text(row["prompt"]), author=chat.Author(chat.AuthorKind.User))]
                spans, parser = renderer.render_for_completion_with_effort(msgs, 0.9)  # as the RL env
                futs.append((row, parser, sampler.sample(prompt=token_spans_to_tinker_model_input(spans),
                                                         sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, stop=renderer.stop()),
                                                         num_samples=args.k)))
            jobs = []
            for row, parser, fut in futs:
                try:
                    res = fut.result(timeout=600)
                except Exception as e:  # noqa: BLE001
                    print(f"  sample failed {row['id']}: {type(e).__name__}", flush=True)
                    continue
                texts = []
                for seq in res.sequences:
                    try:
                        texts.append("\n".join(m.content.text for m in parser.parse_tokens(seq.tokens) if hasattr(m.content, "text")))
                    except Exception:  # noqa: BLE001
                        continue
                texts = list(dict.fromkeys(t.strip() for t in texts))  # identical samples score once
                jobs.append((row, texts, [ex.submit(score_response, row, t, refs.get(row["doc"])) for t in texts]))
            for row, texts, fs in jobs:
                seen += 1
                good, cands = [], []
                for t, f in zip(texts, fs):
                    try:
                        s = f.result(timeout=400)
                    except Exception:  # noqa: BLE001
                        continue
                    cands.append({"text": t, "compiled": s["compiled"], "applied": s["applied"], "sim": s.get("sim"), "strict": s["strict"],
                                  "pages": s.get("pdf_pages"), "pdf_text": (s.get("pdf_text") or "")[:4000] if s["compiled"] else None})
                    if s["compiled"] and (s.get("sim") or 0) >= KEEP_SIM:
                        good.append((t, s["sim"]))
                if dump:
                    dump.write(json.dumps({"id": row["id"], "doc": row["doc"], "cands": cands}) + "\n"); dump.flush()
                rec = {"id": row["id"], "doc": row["doc"], "n_samples": len(texts), "n_exact": len(good)}
                if good:
                    good.sort(key=lambda g: (-g[1], len(g[0])))
                    rec.update(prompt=row["prompt"], target=good[0][0], src=f"ei:{row['id']}")
                    kept += 1
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            el = time.monotonic() - t0
            print(f"[{seen}/{len(todo)}] prompts with an exact sample: {kept} ({kept / max(1, seen):.0%}) {el / 60:.1f}min ETA {el / max(1, seen) * (len(todo) - seen) / 60:.0f}min", flush=True)


if __name__ == "__main__":
    main()
