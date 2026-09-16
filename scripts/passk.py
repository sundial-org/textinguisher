"""Compiler-verified best-of-k at inference: greedy sample + (k-1) samples at T, every candidate compiled; the served answer is the
first candidate that compiles with the fewest deleted content chars (no reference used). Reports the selected answer's compile/exact
rates and the oracle pass@k, on the verified benchmark rows. Writes out/texse2_heldout_<tag>_bo<k>.json in run_eval's format (the response of a retried candidate is both passes joined).
  passk.py --ckpt tinker://.../sampler_weights/000120 --tag rlM6c_000120 --k 4 --temp 0.8"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import common
from fix_env import deleted_chars
from run_eval import score_response


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt2", default=None, help="second checkpoint: the k-1 samples are split between the two policies (mixed pool; the greedy comes from --ckpt)")
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2000); ap.add_argument("--data", default="texse2_heldout.jsonl")
    ap.add_argument("--dump-text", action="store_true", help="with --dump: also store each candidate's text and PDF key (for offline selectors / judges)")
    ap.add_argument("--dump", action="store_true", help="also write every candidate with selector features to out/cands_<tag>_bo<k>.jsonl")
    ap.add_argument("--all", action="store_true", help="score every row of --data, not only the verified benchmark rows")
    ap.add_argument("--retry-conv", action="store_true", help="with --retry: render the second attempt as a continuation of the conversation (system, prompt, first fix, new error) like the RETRY_TURNS RL env, instead of a fresh prompt")
    ap.add_argument("--retry", action="store_true", help="compiler in the loop: every non-compiling candidate gets one second attempt on its own failed state (new error lines + patched doc); the second attempts join the pool")
    ap.add_argument("--verifier", default=None, help="tinker sampler path of a yes/no verifier (build_verifier.py data): select the compiling candidate with the highest P(yes)")
    args = ap.parse_args()
    common.load_env()
    import tinker
    from tinker import types
    from tml_renderers import chat
    from tml_renderers.tinker import token_spans_to_tinker_model_input
    refs = common.load_refs()
    keep = json.loads((common.DATA / "refs2_heldout.json").read_text()); drop = set(json.loads((common.DATA / "texse2_heldout_drop.json").read_text()))
    rows = [json.loads(l) for l in (common.DATA / args.data).read_text().splitlines() if l.strip()]
    rows = rows if args.all else [r for r in rows if r["doc"] in keep and r["id"] not in drop]
    sampler = tinker.ServiceClient().create_sampling_client(model_path=args.ckpt)
    sampler2 = tinker.ServiceClient().create_sampling_client(model_path=args.ckpt2) if args.ckpt2 else None
    renderer = common.get_renderer()

    def render(row):
        msgs = [chat.Message(content=chat.Text(common.system_prompt()), author=chat.Author(chat.AuthorKind.System)),
                chat.Message(content=chat.Text(row["prompt"]), author=chat.Author(chat.AuthorKind.User))]
        return renderer.render_for_completion_with_effort(msgs, 0.9)

    def text_of(parser, seq):
        try:
            return "\n".join(m.content.text for m in parser.parse_tokens(seq.tokens) if isinstance(m.content, chat.Text))
        except Exception:  # noqa: BLE001
            return ""

    verifier = tinker.ServiceClient().create_sampling_client(model_path=args.verifier) if args.verifier else None

    def p_yes(row, fix):
        """P(verifier answers yes): first-token greedy logprob, yes -> p, no -> 1-p, anything else -> 0."""
        from build_verifier import verifier_prompt
        msgs = [chat.Message(content=chat.Text(common.system_prompt()), author=chat.Author(chat.AuthorKind.System)),
                chat.Message(content=chat.Text(verifier_prompt(row["prompt"], fix)), author=chat.Author(chat.AuthorKind.User))]
        spans, parser = renderer.render_for_completion_with_effort(msgs, 0.9)
        return verifier.sample(prompt=token_spans_to_tinker_model_input(spans), sampling_params=types.SamplingParams(max_tokens=4, temperature=0.0, stop=renderer.stop()), num_samples=1)

    def p_of(fut, parser_row):
        import math
        seq = fut.result().sequences[0]
        text = text_of(parser_row, seq).strip().lower()
        lp = seq.logprobs[0] if seq.logprobs else 0.0
        if text.startswith("yes"): return math.exp(lp)
        if text.startswith("no"): return 1.0 - math.exp(lp)
        return 0.0

    dump = open(common.PROJ / "out" / f"cands_{args.tag}_bo{args.k}.jsonl", "w") if args.dump else None
    t0 = time.monotonic(); out = []; n_any = n_sel_comp = n_sel_exact = n_greedy_comp = n_greedy_exact = n_oracle_exact = 0
    with ThreadPoolExecutor(64) as ex:
        for c in range(0, len(rows), 32):
            chunk = rows[c:c + 32]; futs = []
            for row in chunk:
                spans, parser = render(row); mi = token_spans_to_tinker_model_input(spans)
                fg = sampler.sample(prompt=mi, sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=0.0, stop=renderer.stop()), num_samples=1)
                n1 = (args.k - 1) // 2 if sampler2 else args.k - 1
                fs = sampler.sample(prompt=mi, sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=args.temp, stop=renderer.stop()), num_samples=n1) if n1 else None
                fs2 = sampler2.sample(prompt=mi, sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=args.temp, stop=renderer.stop()), num_samples=args.k - 1 - n1) if sampler2 else None
                futs.append((row, parser, fg, fs, fs2))
            jobs = []
            for row, parser, fg, fs, fs2 in futs:
                texts = [text_of(parser, fg.result().sequences[0])] + ([text_of(parser, s) for s in fs.result().sequences] if fs else []) + ([text_of(parser, s) for s in fs2.result().sequences] if fs2 else [])
                jobs.append((row, parser, texts, [ex.submit(score_response, row, t, refs.get(row["doc"])) for t in texts]))
            for row, parser, texts, fs_ in jobs:
                scored = [f.result() for f in fs_]
                dels = [deleted_chars(t) for t in texts]
                if args.retry:  # second pass for the failures, sampled at the same temperature, scored from the failed candidate's document
                    rfuts, rinfo = [], []
                    for i, sc in enumerate(scored):
                        if sc["compiled"] or not sc.get("retry_doc"):
                            continue
                        rp = common.build_fix_prompt("main.tex", sc["retry_error_lines"], sc["retry_log_tail"], sc["retry_doc"], full_lines=500, max_chars=16000)
                        if args.retry_conv:
                            msgs2 = [chat.Message(content=chat.Text(common.system_prompt()), author=chat.Author(chat.AuthorKind.System)),
                                     chat.Message(content=chat.Text(row["prompt"]), author=chat.Author(chat.AuthorKind.User)),
                                     chat.Message(content=chat.Text(texts[i]), author=chat.Author(chat.AuthorKind.Model)),
                                     chat.Message(content=chat.Text(rp), author=chat.Author(chat.AuthorKind.User))]
                            spans2, parser2 = renderer.render_for_completion_with_effort(msgs2, 0.9)
                        else:
                            spans2, parser2 = render({**row, "prompt": rp})
                        rfuts.append(sampler.sample(prompt=token_spans_to_tinker_model_input(spans2), sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=args.temp if i else 0.0, stop=renderer.stop()), num_samples=1))
                        rinfo.append((i, parser2, sc["retry_doc"]))
                    r_texts = [text_of(pr, f.result().sequences[0]) for f, (_, pr, _) in zip(rfuts, rinfo)]
                    r_scored = [f.result() for f in [ex.submit(score_response, row, t, refs.get(row["doc"]), sd) for t, (_, _, sd) in zip(r_texts, rinfo)]]
                    for (i, _, _), t, sc2 in zip(rinfo, r_texts, r_scored):
                        texts.append(texts[i] + "\n\n" + t); scored.append(sc2); dels.append(dels[i] + deleted_chars(t))
                comp = [i for i, s in enumerate(scored) if s["compiled"]]
                # consensus: candidates whose PDFs agree (same pages, same text) vote together; fewest deleted chars breaks ties
                key = lambda i: (scored[i].get("pdf_pages"), " ".join((scored[i].get("pdf_text") or "").split()))
                votes = {}
                for i in comp:
                    votes[key(i)] = votes.get(key(i), 0) + 1
                # kept-first: candidates that delete <= 40 content chars rank ahead (deletion hacks compile but are not fixes); then the
                # verifier's P(yes) if one is given (unrounded: it saturates near 0/1 but tiny margins still rank), then consensus votes
                pv = {}
                if verifier and comp:
                    vf = {i: p_yes(row, texts[i]) for i in comp}
                    pv = {i: p_of(vf[i], parser) for i in comp}
                sel = min(comp, key=lambda i: (dels[i] > 40, -pv.get(i, 0.0), -votes[key(i)], dels[i], i)) if comp else 0
                n_oracle_exact += any(s["strict"] for s in scored)
                if dump:
                    dump.write(json.dumps({"id": row["id"], "cands": [{"i": i, "compiled": sc["compiled"], "strict": sc["strict"], "deleted": dels[i], "retry": i >= args.k,
                                "diff_lines": sc.get("diff_lines"), "pages": sc.get("pdf_pages"), "votes": votes.get(key(i), 0) if sc["compiled"] else 0,
                                "len": len(texts[i]), "n_edits": texts[i].count("<edit>"), "greedy": i == 0, "p_yes": pv.get(i),
                                "text": texts[i] if args.dump_text else None, "pdf_key": (sc.get("pdf_pages"), (sc.get("pdf_text") or "")[:3000]) if (args.dump_text and sc["compiled"]) else None} for i, sc in enumerate(scored)]}) + "\n"); dump.flush()
                s = scored[sel]
                n_any += bool(comp); n_sel_comp += s["compiled"]; n_sel_exact += s["strict"]; n_greedy_comp += scored[0]["compiled"]; n_greedy_exact += scored[0]["strict"]
                out.append({"id": row["id"], "response": texts[sel], "selected": sel, "n_compiling": len(comp), "votes": votes.get(key(sel), 0) if comp else 0, "out_tokens": None, "wall_s": None,
                            **{k: s.get(k) for k in ("applied", "compiled", "exact", "diff_lines", "pdf_sim", "text_sim", "img_sim", "sim", "pages_ok", "strict", "pdf_pages")}})
            n = len(out)
            print(f"[{n}/{len(rows)}] greedy {n_greedy_comp / n:.3f}/{n_greedy_exact / n:.3f}  selected {n_sel_comp / n:.3f}/{n_sel_exact / n:.3f}  oracle pass@{args.k} {n_any / n:.3f}/{n_oracle_exact / n:.3f}  {(time.monotonic() - t0) / 60:.1f}min", flush=True)
    json.dump({"data": f"data/{args.data}", "arm": args.ckpt, "bo": args.k, "temp": args.temp, "verifier": args.verifier, "retry": args.retry, "ckpt2": args.ckpt2, "results": out}, open(common.PROJ / "out" / f"{args.data.rsplit('.', 1)[0]}_{args.tag}_bo{args.k}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
