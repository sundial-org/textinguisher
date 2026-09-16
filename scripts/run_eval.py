"""Evaluate a model on fix examples: sample -> parse edit blocks -> apply -> recompile.
Success = the patched project compiles clean. Also records wall-clock and output size.

  python3 run_eval.py --data ../data/eval.jsonl --arm inkling-small
  python3 run_eval.py --data ../data/eval.jsonl --arm tinker://<checkpoint path>
  python3 run_eval.py --data ../data/eval.jsonl --arm gateway:anthropic/claude-haiku-4.5
"""
import argparse
import asyncio
import difflib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common

COMPILE_TIMEOUT = int(common.os.environ.get("COMPILE_TIMEOUT", "90"))  # RL lowers it: one straggler stalls a whole batch


def broken_text(row: dict) -> str:
    doc = (Path(row["src_dir"]) / "main.tex").read_text(errors="replace")
    for e in row.get("break_edits") or ([row["break_edit"]] if row.get("break_edit") else []):
        doc = doc.replace(e["search"], e["replace"], 1)  # no edits = organic failure, doc is already broken
    return doc


def broken_project(row: dict) -> dict[str, str]:
    files = common.project_files(row["src_dir"])
    for e in row.get("file_edits", []):
        files[e["file"]] = files[e["file"]].replace(e["search"], e["replace"], 1)
    return files


def score_project(row: dict, response: str, ref: dict | None = None) -> dict:
    """v3 project lane: file-aware edit blocks applied to a multi-file project."""
    files = broken_project(row)
    edits = common.parse_file_edits(response)
    patched, err = common.apply_file_edits(files, edits)
    out = {"parsed": bool(edits), "applied": err is None, "apply_error": err, "compiled": False,
           "exact": False, "diff_lines": None, "pdf_sim": None, "pages_ok": None, "strict": False}
    if err is None:
        orig = common.project_files(row["src_dir"])
        labeled = bool(row.get("file_edits"))  # src_dir holds the human-fixed project only for labeled rows
        out["exact"] = patched == orig and labeled
        out["diff_lines"] = sum(len([l for l in difflib.unified_diff(files[f].splitlines(), patched[f].splitlines(), lineterm="")
                                     if l[:1] in "+-" and l[:3] not in ("+++", "---")]) for f in patched if patched[f] != files[f])
        if out["exact"]:
            out.update(compiled=True, pdf_sim=1.0, pages_ok=True, strict=True, text_sim=1.0, img_sim=1.0, sim=1.0)
        elif patched == orig:  # organic row (no human fix): a no-op edit leaves the known-broken project; no compile needed
            pass
        else:
            overrides = {f: patched[f] for f in patched if patched[f] != orig.get(f)}
            r = common.compile_project(Path(row["src_dir"]), [], overrides=overrides, timeout=COMPILE_TIMEOUT,
                                       bib=row.get("bib", False))
            out["compiled"], out["timed_out"], out["texlive"] = r["ok"], bool(r.get("timed_out")), r.get("texlive")
            if r["ok"]:
                out["pdf_pages"], out["pdf_text"] = r.get("pdf_pages"), r.get("pdf_text")  # lets a selector compare candidates' outputs
            if not r["ok"]:  # what a retry turn needs (the project lane retries on the reported file only)
                out["retry_doc"] = patched.get(row["reported_file"], "")
                out["retry_error_lines"] = common.error_lines_from_items(r["items"], row["reported_file"], set(patched))
                out["retry_log_tail"] = r["log"][-common.MAX_LOG_CHARS:]
            if r["ok"] and ref:
                out.update(common.pdf_score(r, ref))
    return out


def score_response(row: dict, response: str, ref: dict | None = None,
                   start_doc: str | None = None) -> dict:
    response = common.THINK_RE.sub("", response, count=1)  # thinking-budget arms reason before the edits
    if row.get("lane") == "project":
        return score_project(row, response, ref)
    doc = start_doc if start_doc is not None else broken_text(row)
    edits = common.parse_edits(response) or [(s, r) for _, s, r in common.parse_file_edits(response)]  # single-file rows: accept <edit file=...> too
    patched, apply_err = common.apply_edits(doc, edits)
    out = {"parsed": bool(edits), "applied": apply_err is None, "apply_error": apply_err,
           "compiled": False, "exact": False, "diff_lines": None,
           "pdf_sim": None, "pages_ok": None, "strict": False}
    if apply_err is None:
        orig = (Path(row["src_dir"]) / "main.tex").read_text(errors="replace")
        out["exact"] = patched == orig and bool(row.get("break_edits") or row.get("break_edit"))
        out["diff_lines"] = sum(
            1 for l in difflib.unified_diff(doc.splitlines(), patched.splitlines(), lineterm="")
            if l[:1] in "+-" and l[:3] not in ("+++", "---"))
        if out["exact"]:
            out.update(compiled=True, pdf_sim=1.0, pages_ok=True, strict=True, text_sim=1.0, img_sim=1.0, sim=1.0)
        else:
            r = common.compile_project(
                Path(row["src_dir"]), [Path(b) for b in row["bundles"]],
                overrides={"main.tex": patched}, timeout=COMPILE_TIMEOUT)
            out["compiled"], out["timed_out"], out["texlive"] = r["ok"], bool(r.get("timed_out")), r.get("texlive")
            if r["ok"]:
                out["pdf_pages"], out["pdf_text"] = r.get("pdf_pages"), r.get("pdf_text")  # lets a selector compare candidates' outputs
            if not r["ok"]:  # keep what a retry needs: the patched doc and its fresh error state
                project_files = {p.name for p in Path(row["src_dir"]).iterdir()}
                out["retry_doc"] = patched
                out["retry_error_lines"] = common.error_lines_from_items(r["items"], "main.tex", project_files)
                out["retry_log_tail"] = r["log"][-common.MAX_LOG_CHARS:]
            if r["ok"] and ref:
                out.update(common.pdf_score(r, ref))
    return out


def sample_tinker(rows: list[dict], model: str, sequential: bool = False) -> list[dict]:
    import tinker
    from tinker import types
    from tml_renderers import chat
    from tml_renderers.tinker import token_spans_to_tinker_model_input

    sc = tinker.ServiceClient()
    if model.startswith("tinker://"):
        sampler = sc.create_sampling_client(model_path=model)
    else:
        sampler = sc.create_sampling_client(base_model=model)
    if common.RENDERER != "tml_v0":  # non-Inkling base: cookbook renderer for prompt building and parsing
        cr = common.cookbook_renderer()
        max_tokens = int(common.os.environ.get("EVAL_MAX_TOKENS", 2000)) + common.THINK_BUDGET

        def submit_cb(row):
            t0 = time.monotonic()
            fut = sampler.sample(prompt=cr.build_generation_prompt(common.cookbook_messages(row["prompt"])),
                                 sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=float(common.os.environ.get("EVAL_TEMP", "0")), stop=cr.get_stop_sequences()), num_samples=1)
            return t0, fut

        def collect_cb(t0, fut):
            try:
                seq = fut.result().sequences[0]
                msg, _ = cr.parse_response(seq.tokens)
                return {"response": common.cookbook_text(msg), "wall_s": round(time.monotonic() - t0, 2), "out_tokens": len(seq.tokens), "sample_error": None}
            except Exception as e:  # noqa: BLE001
                return {"response": "", "wall_s": None, "out_tokens": None, "sample_error": str(e)[:200]}

        if sequential:
            return [collect_cb(*submit_cb(r)) for r in rows]
        futs = [submit_cb(r) for r in rows]
        return [collect_cb(t0, f) for t0, f in futs]
    renderer = common.get_renderer()

    def submit(row):
        msgs = [
            chat.Message(content=chat.Text(common.system_prompt()), author=chat.Author(chat.AuthorKind.System)),
            chat.Message(content=chat.Text(row["prompt"]), author=chat.Author(chat.AuthorKind.User)),
        ]
        # same prompt as the RL env: the cookbook tml_v0 renderer conditions on a reasoning effort (default 0.9)
        spans, parser = renderer.render_for_completion_with_effort(msgs, float(common.os.environ.get("EVAL_EFFORT", "0.9")))
        fut = sampler.sample(
            prompt=token_spans_to_tinker_model_input(spans),
            sampling_params=types.SamplingParams(
                max_tokens=int(common.os.environ.get("EVAL_MAX_TOKENS", 2000)) + common.THINK_BUDGET, temperature=float(common.os.environ.get("EVAL_TEMP", "0")), stop=renderer.stop()),
            num_samples=1)
        return parser, fut

    def collect(t0, parser, fut):
        try:
            seq = fut.result().sequences[0]
            text = "\n".join(m.content.text for m in parser.parse_tokens(seq.tokens)
                             if isinstance(m.content, chat.Text))
            return {"response": text, "wall_s": round(time.monotonic() - t0, 2),
                    "out_tokens": len(seq.tokens), "sample_error": None}
        except Exception as e:  # noqa: BLE001
            return {"response": "", "wall_s": None, "out_tokens": None,
                    "sample_error": str(e)[:200]}

    if sequential:  # true per-request latency
        samples = []
        for row in rows:
            t0 = time.monotonic()
            parser, fut = submit(row)
            samples.append(collect(t0, parser, fut))
        return samples
    t0 = time.monotonic()
    futures = [submit(row) for row in rows]  # batch throughput; wall_s is not per-request latency
    return [collect(t0, parser, fut) for parser, fut in futures]


async def sample_gateway(rows: list[dict], model: str, concurrency: int, temperature: float = 0.0) -> list[dict]:
    import openai

    client = openai.AsyncOpenAI(base_url="https://ai-gateway.vercel.sh/v1", max_retries=2,  # a hung gateway call must not stall the run
                                timeout=float(common.os.environ.get("GATEWAY_TIMEOUT", "600")),
                                api_key=common.os.environ["AI_GATEWAY_API_KEY"])
    sem = asyncio.Semaphore(concurrency)

    async def one(row):
        async with sem:
            t0 = time.monotonic()
            try:
                resp = await client.chat.completions.create(
                    model=model, max_tokens=int(common.os.environ.get("EVAL_MAX_TOKENS", 2000)), temperature=temperature,
                    messages=[{"role": "system", "content": common.system_prompt()},
                              {"role": "user", "content": row["prompt"]}])
                text = resp.choices[0].message.content or ""
                out_tokens = resp.usage.completion_tokens if resp.usage else None
                err = None
            except Exception as e:  # noqa: BLE001
                text, out_tokens, err = "", None, str(e)[:200]
            return {"response": text, "wall_s": round(time.monotonic() - t0, 2),
                    "out_tokens": out_tokens, "sample_error": err}

    return await asyncio.gather(*[one(r) for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.DATA / "eval.jsonl"))
    ap.add_argument("--arm", required=True,
                    help="inkling-small | tinker://<path> | thinkingmachines/<model> | gateway:<slug>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--sequential", action="store_true", help="one request at a time (latency bench)")
    ap.add_argument("--retries", type=int, default=0, help="1 = pass@2 with recompile feedback")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    common.load_env()
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    arm = "tinker:thinkingmachines/Inkling-Small" if args.arm == "inkling-small" else args.arm
    print(f"{len(rows)} examples, arm={arm}")

    model = arm.split(":", 1)[1] if arm.startswith("tinker:") and not arm.startswith("tinker://") else arm
    if arm.startswith("gateway:"):
        conc = 1 if args.sequential else args.concurrency
        samples = asyncio.run(sample_gateway(rows, arm.split(":", 1)[1], conc))
    else:
        samples = sample_tinker(rows, model, sequential=args.sequential)

    refs = common.load_refs()
    results = []
    with ProcessPoolExecutor(6) as ex:
        futs = [ex.submit(score_response, row, s["response"], refs.get(row["doc"]))
                for row, s in zip(rows, samples)]
        for row, s, f in zip(rows, samples, futs):
            results.append({"id": row["id"], "mutation": row["mutation"], **s, **f.result()})

    if args.retries:  # pass@2 with the compiler in the loop (prod retries the fix turn too)
        todo = [(i, r) for i, r in enumerate(results) if not r["compiled"]]
        retry_rows = []
        for i, r in todo:
            row = rows[i]
            if r.get("retry_doc"):
                v2 = "break_edits" in row
                prompt = common.build_fix_prompt(
                    "main.tex", r["retry_error_lines"], r["retry_log_tail"], r["retry_doc"],
                    full_lines=500 if v2 else common.MAX_FULL_SOURCE_LINES,
                    max_chars=16000 if v2 else common.MAX_SOURCE_CHARS)
            else:  # edit did not apply: same prompt, plus what went wrong
                prompt = row["prompt"] + ("\n\nYour previous edit could not be applied: "
                                          f"{r['apply_error']}. The SEARCH text must match the file exactly.")
            retry_rows.append({**row, "prompt": prompt})
        print(f"\nretrying {len(retry_rows)} failures...")
        if arm.startswith("gateway:"):
            rs = asyncio.run(sample_gateway(retry_rows, arm.split(":", 1)[1], args.concurrency))
        else:
            rs = sample_tinker(retry_rows, model)
        with ProcessPoolExecutor(6) as ex:
            futs = [ex.submit(score_response, rows[i], s["response"], refs.get(rows[i]["doc"]),
                              results[i].get("retry_doc"))
                    for (i, _), s in zip(todo, rs)]
            for (i, _), s, f in zip(todo, rs, futs):
                sc = f.result()
                results[i]["pass1"] = False
                results[i].update({"compiled": sc["compiled"], "strict": sc["strict"],
                                   "retry_response": s["response"], "retry_applied": sc["applied"]})
    for r in results:  # strip bulky retry state from the saved results
        r.setdefault("pass1", r["compiled"])
        r.pop("retry_doc", None); r.pop("retry_log_tail", None); r.pop("retry_error_lines", None)

    n = len(results)
    fixed = sum(r["compiled"] for r in results)
    strict = sum(r["strict"] for r in results)
    exact = sum(r["exact"] for r in results)
    applied = sum(r["applied"] for r in results)
    walls = sorted(r["wall_s"] for r in results if r["wall_s"] is not None)
    toks = [r["out_tokens"] for r in results if r["out_tokens"]]
    if args.retries:
        p1 = sum(r["pass1"] for r in results)
        print(f"\npass@1:          {p1}/{n} = {p1 / n:.1%}")
    print(f"\nfixed(compiles): {fixed}/{n} = {fixed / n:.1%}" + ("  (pass@2)" if args.retries else ""))
    if refs:
        print(f"fixed strict:    {strict}/{n} = {strict / n:.1%}  (compiles + PDF matches ref)")
    print(f"exact-match:     {exact}/{n} = {exact / n:.1%}")
    print(f"edits applied:   {applied}/{n}")
    if walls:
        print(f"wall s: p50={walls[len(walls) // 2]:.2f} p90={walls[int(len(walls) * 0.9)]:.2f}")
    if toks:
        print(f"out tokens: mean={sum(toks) / len(toks):.0f}")
    by_mut: dict[str, list] = {}
    for r in results:
        by_mut.setdefault(r["mutation"], []).append(r["compiled"])
    for m, vals in sorted(by_mut.items()):
        print(f"  {m:18s} {sum(vals)}/{len(vals)}")

    slug = arm.replace("/", "_").replace(":", "_").replace(".", "-")
    out = Path(args.out) if args.out else common.PROJ / "out" / f"eval_{slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arm": arm, "data": args.data, "n": n, "fixed": fixed, "retries": args.retries,
                               "strict": strict, "exact": exact, "results": results}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
