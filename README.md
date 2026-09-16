# textinguisher

Fixing real LaTeX compile errors in under a second: a fine-tuned Inkling-Small, TeXtinguisher (the benchmark it was
measured on), and the pipeline that produced it. Companion to the post *Fixing LaTeX errors in under a second* (Sundial blog).

## Why

AI is moving mathematics quickly, yet most mathematicians' day-to-day tools have changed little in a decade, and one of
the complaints we heard most often when talking to them was fighting LaTeX compilation errors: writing gets interrupted
by a missing brace or two packages that clash, the log is two hundred lines long, and the flow is gone. Asking a chatbot
works, but switching windows breaks the flow too, and a pasted snippet loses the context that lives in other files.

We wanted a model that fixes the error inside the editor, in well under a second, as a suggestion that can be reverted
in a click, cheap enough to run on every compile. That rules out frontier APIs (several seconds, a page of reasoning per
fix) and points at a small open model fine-tuned for one job: Thinking Machines' Inkling-Small through Tinker.

## What is here

- A benchmark of real compile errors mined from TeX.StackExchange (asker's document still fails, accepted answer's
  document compiles, diff replays exactly), plus real multi-file projects from GitHub histories and old arXiv sources that
  no longer build under a current TeX Live. Every row is compile-verified; the released subset is in `release/`.
- The training recipe: supervised fine-tuning on real fixes, then GRPO with the compiler as the reward (a fix that compiles
  is rewarded, more if the PDF matches the human fix, nothing once it deletes content), with companion files in the prompt
  for multi-file projects.
- The evaluation harness, the Modal compile farm, and the merge/serve/bench scripts.

Result on 647 held-out real errors (669 mined, 22 dropped after an audit found their reference broken): the fine-tuned model fixes
85.2% (GPT-6 Astra 90.9%, Claude Fable 5.1 82.4%, GPT-5.5 80.7%) in 0.95 s per fix at about $1.30 per thousand fixes; the margin over
GPT-5.5 holds under a paired McNemar test (p = 0.02), Fable is within noise (p = 0.15), and Astra is ahead (p = 0.001) at fourteen
times the cost. The frontier models still delete less content when they fix, and still lead
on real multi-file projects; both are documented in the post and in `release/DATASET_CARD.md`.

## Layout

- `scripts/` pipeline (`python` below = `../.venv/bin/python`, run from the repo root; data paths resolve through
  `common.PROJ`, so any cwd works).
  `common.py` ports sundial's `latex-log-parser.ts` + `fix-prompt.ts` (`reference/`) so prompts match the product.
- `modal/` compile farm (`modal_compile.py`, `modal_compile_hist.py`), LoRA merge (`merge_inkling.py`), vLLM serving
  (`modal_vllm_merged.py`) and benches (`bench.py`, `bench_throughput.py`).
- `data/` (internal, not shipped) jsonl rows (`prompt`, `target`, `src_dir`, `break_edits`/`file_edits`, provenance). Frozen held-out sets:
  `texse_heldout` (287), `texse2_heldout` (669 mined, 647 after the audit; drop list `texse2_heldout_drop.json`), `multi2_heldout`, `skew_multi_heldout`, `skew_eval`, `project_eval`,
  `hard_eval3`, `eval` (v1). `refs.json` = PDF text/pages of the human fix, `refs_pool.json` = teacher pseudo-refs,
  `refs_skew.json` = historic-TeX-Live renders. TeX.SE rows are CC BY-SA (`license` per row); arXiv/GitHub rows keep
  ids, sha and licence fields. `*_report.md` describe each mining run.
- `release/` public subset (`scripts/build_release.py`, `release/DATASET_CARD.md`): TeX.SE and licensed GitHub rows with
  sources; skew rows are metadata-only and `refs_skew.json` is not shipped.
- `out/` eval results `<set>_<tag>.json`, RL logs `rl_*/metrics.jsonl`, chain scripts, figures.

## Pipeline

```
# mine (COMPILE_BACKEND=modal MODAL_PROFILE=<profile> for the farm; GitHub miners compile locally otherwise)
python scripts/mine_texse.py --target 400; python scripts/mine_texse2.py --resume; python scripts/mine_texse_pool.py --resume
python scripts/mine_github_multi.py scan|mine|verify; python scripts/mine_github_multi2.py repos|walk
python scripts/mine_github_forks.py forks|walk; python scripts/mine_texse_multi.py parse|mine
python scripts/gen_skew_train.py --target 3000 --resume            # organic arXiv version skew, historic refs
python scripts/gen_arxiv.py; python scripts/gen_data.py; python scripts/gen_project.py; python scripts/gen_multi_synth.py   # synthetic sets
# split (frozen files; never re-draw after a checkpoint exists)
python scripts/split_texse.py --jsonl texse; python scripts/split_texse.py --data data/texse2_labeled.jsonl --out data/texse2_split.json --probed --jsonl texse2
python scripts/finish_multi2.py; python scripts/gen_skew_train.py --split; python scripts/split_data.py
# SFT
python scripts/distill.py --data data/texse_train.jsonl --teacher gateway:anthropic/claude-fable-5.1 --k 2
python scripts/build_pseudo_refs.py; python scripts/build_round5.py --mix <file:weight,...> --out train6.jsonl
python scripts/train_sft.py --data data/train6.jsonl --val-data data/val6.jsonl --epochs 3
python scripts/build_multi_sft.py; python scripts/train_sft.py --data data/multi_sft.jsonl --load-state tinker://.../weights/final --epochs 1
# RL (state path from out/checkpoints.txt)
COMPILE_BACKEND=modal COMPILE_TIMEOUT=40 python scripts/train_rl.py mode=fix load_checkpoint_path=tinker://.../weights/state2 train=<file:weight,...> log_path=out/rl_fix
# eval
scripts/eval_battery.sh <arm> <tag> [sets...]        # arm = tinker://<sampler path> | inkling-small | gateway:<slug>
python scripts/run_eval.py --data data/<set>.jsonl --arm <arm> --out out/<set>_<tag>.json
python scripts/heldout_table.py texse2_heldout; python scripts/select_ckpt.py rlM2 rlM3; python scripts/refill_empty.py out/<set>_<tag>.json
# merge + bench (out/finalize.sh <tag> <sampler path> runs all of this)
modal run modal/merge_inkling.py --stage adapter|aux|merge|verify; modal deploy modal/modal_vllm_merged.py
python modal/bench.py <url> --model merged --n 15; python modal/bench_throughput.py <url> --concurrency 64 --n 320
```

## Environment

| var | read by | meaning (default) |
|---|---|---|
| `TML_KEY`, `AI_GATEWAY_API_KEY` | `common.load_env` (`../.env` or `.env`) | Tinker key (exported as `TINKER_API_KEY`); Vercel AI Gateway key for baseline arms |
| `COMPILE_BACKEND=modal`, `MODAL_PROFILE` | `common.compile_project` | compile on the Modal farm `latexfix-compile` instead of local latexmk; rows with `bundles` stay local |
| `COMPILE_SLOTS` | `agent_env` (RL/eval) | concurrent local compiles (6) |
| `COMPILE_TIMEOUT` | `run_eval`, RL | seconds per compile (90; RL runs used 40) |
| `COMPILE_W STRICT_W MINIMAL_W STRUCTURE_W LENGTH_W` | `fix_env` | reward weights (1.0 / 0.5 / 0 / 0 / 0); the model of record uses the defaults |
| `THINK_BUDGET`, `MAX_GEN_TOKENS` | `common.system_prompt`, `fix_env` | reasoning allowance in the prompt (0) and RL generation cap (1500) |
| `EVAL_MAX_TOKENS` | `run_eval`, `refill_empty` | eval generation cap (2000); Tinker arms add `THINK_BUDGET` |
| `GATE287`, `TP_FILE` | `select_ckpt`, `plot_cost` | texse_heldout gate for checkpoint ranking (70.4); throughput bench json |

Local runs need latexmk (TeX Live) and poppler (`pdftotext`, `pdfinfo`) on PATH, `tinker_cookbook` with the `tml_v0`
renderer, and the `tml_renderers` partner wheel.

## Metrics

**compiles**: the patched project builds clean. **<=40 removed**: compiles and the edits remove at most 40 non-comment
characters (catches "delete or comment out until it compiles"). **strict**: compiles and the PDF text/page count match the
reference (similarity >= 0.985; skew refs use 0.93 on ASCII text). `heldout_table.py` also prints **intact** (nothing
removed, no structural construct lost).

## Limitations

The deletion metric is a heuristic: it counts characters removed inside SEARCH/REPLACE blocks or commented out, so
`\iffalse`, comment environments and macro stubs escape it. The compile farm image is `texlive/texlive:latest`, which was
TeX Live 2026 at the time of the final evaluations, while local compiles and the older refs used TeX Live 2025. Strict
compares against the accepted answer's document, which is often restyled, so valid alternative fixes score 0. Checkpoints
were selected on separate dev sets (`texse_heldout`, `multi2_heldout`, `skew_multi_heldout`) that share no questions
with the 647-row `texse2_heldout` headline set; numbers reported on the selection sets are optimistic.
