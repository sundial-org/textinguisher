"""GRPO (cookbook rl.train, importance-sampling loss, group-centered advantages) for the single-shot LaTeX-fix env.

  train_rl.py mode=fix load_checkpoint_path=tinker://.../weights/state2 log_path=out/rl_fix
`train` is a comma list of data/<file>:<weight> (weight 2 = repeat twice, 0.5 = sample half); rows whose
id is in `val` are excluded. The mix is written to data/rl_<run>_train.jsonl. Progress: <log_path>/metrics.jsonl (reward, env/compiled, env/strict...).
"""
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train

import common

HELDOUT_FILES = ("texse_heldout.jsonl", "texse2_heldout.jsonl", "multi2_heldout.jsonl", "skew_multi_heldout.jsonl", "skew_eval.jsonl", "project_eval.jsonl")
MAX_PROMPT_CHARS = 34000  # context prompts for multi-file projects run to ~33k chars (~10k tokens)
BAD_RE = re.compile(r"\ufffd|\.(cls|sty)' not found|cref@override@label@type|caption@setfloatcapt|"
                    r"Command \\[a-zA-Z@]+ already defined.*\n.*Command \\[a-zA-Z@]+ already defined.*\n.*already defined", re.I)


@chz.chz
class CLI:
    mode: str = "fix"
    train: str = "texse_train_rl.jsonl:1,texse_pool.jsonl:1,project_train_all.jsonl:0.5"
    val: str | None = "rl_val.jsonl"
    load_checkpoint_path: str | None = None
    batch_size: int = 16
    group_size: int = 8
    lr: float = 2e-5
    max_tokens: int = 1500
    log_path: str = "out/rl_fix"
    eval_every: int = 10
    save_every: int = 10
    max_steps: int | None = None
    kl_penalty_coef: float = 0.0
    lora_rank: int = 32
    seed: int = 0
    skip_rows: int = 0
    require_ref: bool = False  # drop rows whose doc has no reference in common.load_refs() (REFS env)
    drop_bad: bool = False  # drop prompts the audit found unfixable: U+FFFD from decoding, missing .cls/.sty, class-version breakage
    async_steps: int = 0  # >0: overlap sampling and training (samples may be this many steps stale)
    kl_ref: str | None = None  # sampler path of a frozen reference policy; kl_penalty_coef > 0 keeps the policy near it
    teacher_w: float = 0.0  # >0 with TEACHER_DEMOS: per-token cross-entropy weight of queued teacher fixes added to every optimizer step
    teacher_max: int = 16  # at most this many teacher fixes per step
    sdft_w: float = 0.0  # >0 with TEACHER_DEMOS: demonstration-conditioned self-distillation (frozen sdft_ref sees the verified fix; its
    sdft_ref: str | None = None  # log-prob of the student's sampled tokens minus the sampled log-prob, clipped +-5, times sdft_w, is added per token)
    wandb_project: str | None = None


def build_train_file(cli: CLI) -> Path:
    rng = random.Random(cli.seed)
    val_ids = {json.loads(l)["id"] for l in (common.DATA / cli.val).read_text().splitlines()} if cli.val else set()
    out, counts = [], {}
    refs = common.load_refs() if cli.require_ref else None
    for spec in cli.train.split(","):
        name, _, w = spec.partition(":")
        w = float(w or 1)
        rows = [json.loads(l) for l in (common.DATA / name).read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r["id"] not in val_ids and len(r["prompt"]) <= MAX_PROMPT_CHARS]
        if refs is not None:
            n0, rows = len(rows), [r for r in rows if r["doc"] in refs]
            print(f"  {name}: {n0 - len(rows)} rows without reference dropped ({len(rows)} kept)", flush=True)
        if cli.drop_bad:
            n0, rows = len(rows), [r for r in rows if not BAD_RE.search(r["prompt"])]
            print(f"  {name}: {n0 - len(rows)} unfixable-signature rows dropped ({len(rows)} kept)", flush=True)
        keep = rows * int(w) + [r for r in rows if rng.random() < w - int(w)]
        counts[name] = len(keep)
        out += keep
    held = {json.loads(l)["id"] for f in HELDOUT_FILES if (common.DATA / f).exists()
            for l in (common.DATA / f).read_text().splitlines() if l.strip()}
    leaked = {r["id"] for r in out} & held
    assert not leaked, f"{len(leaked)} held-out ids in the train mix, e.g. {sorted(leaked)[:3]}"
    rng.shuffle(out)
    path = common.DATA / f"rl_{Path(cli.log_path).name}_train.jsonl"  # one mix file per run: parallel launches must not race
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"train mix {counts} -> {len(out)} rows, {len(out) // cli.batch_size} steps/epoch -> {path}", flush=True)
    return path


def install_teacher_hook(w: float, cap: int) -> None:
    """Before every optimizer step, add a cross-entropy forward/backward on the teacher fixes fix_env queued for prompts whose
    whole group missed the reference (weights scaled by w). Gradients accumulate with the RL step's, so this is one update."""
    import tinker
    import fix_env
    from train_sft import example_to_datum
    renderer = common.get_renderer() if common.RENDERER == "tml_v0" else common.cookbook_renderer()
    orig = tinker.TrainingClient.optim_step_async
    n_total = [0]

    async def optim_step_async(self, *a, **k):
        batch, fix_env.TEACHER_QUEUE[:] = fix_env.TEACHER_QUEUE[:cap], fix_env.TEACHER_QUEUE[cap:]
        if batch:
            datums = []
            for ex in batch:
                d = example_to_datum(ex, renderer)
                d.loss_fn_inputs["weights"] = tinker.TensorData.from_torch(d.loss_fn_inputs["weights"].to_torch() * w)
                datums.append(d)
            fut = await self.forward_backward_async(datums, loss_fn="cross_entropy")
            n_total[0] += len(batch)
            print(f"teacher: {len(batch)} fixes this step ({n_total[0]} total, {len(fix_env.TEACHER_QUEUE)} queued)", flush=True)
        return await orig(self, *a, **k)

    tinker.TrainingClient.optim_step_async = optim_step_async


PRIV = "\n\nFor reference, a verified fix for this error is:\n{demo}\nJudge the proposed edits against it."


def install_sdft_hook(w: float, ref_path: str, clip: float = 5.0) -> None:
    """Demonstration-conditioned self-distillation: a frozen copy of the start policy that is shown the verified fix scores the student's
    sampled tokens; clip(log q - log pi_old) * w is added to each action token's advantage (the environment advantage stays)."""
    import tinker
    import torch
    import fix_env
    from tinker_cookbook.rl import train as cb_train
    ref = tinker.ServiceClient().create_sampling_client(model_path=ref_path)
    renderer = common.cookbook_renderer()
    orig = cb_train.assemble_training_data

    def patched(groups, advantages):
        data_D, meta_D = orig(groups, advantages)
        jobs, registered = [], len(fix_env.SDFT_ROWS)
        for d, m in zip(data_D, meta_D):
            traj = groups[m["group_idx"]].trajectories_G[m["traj_idx"]]
            if len(traj.transitions) != 1:
                continue
            ac = traj.transitions[0].ac
            info = fix_env.SDFT_ROWS.pop(tuple(int(x) for x in ac.tokens), None)  # rollouts run ahead of training: never clear the registry
            if info is None:
                continue
            prefix = renderer.build_generation_prompt(common.cookbook_messages(info[0] + PRIV.format(demo=info[1])))
            full = tinker.ModelInput(chunks=list(prefix.chunks) + [tinker.types.EncodedTextChunk(tokens=list(ac.tokens))])
            jobs.append((d, ac, prefix.length, ref.compute_logprobs(full)))
        n, tot = 0, 0.0
        for d, ac, plen, fut in jobs:
            try:
                lp = fut.result()[plen:plen + len(ac.tokens)]
            except Exception as e:  # noqa: BLE001
                print(f"sdft: teacher scoring failed: {type(e).__name__}", flush=True); continue
            if len(lp) != len(ac.tokens) or any(x is None for x in lp):
                continue
            delta = torch.clamp(torch.tensor(lp) - torch.tensor(ac.logprobs), -clip, clip) * w
            adv = d.loss_fn_inputs["advantages"].to_torch().clone()
            adv[-len(ac.tokens):] += delta
            d.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(adv)
            n += 1; tot += float(delta.mean())
        print(f"sdft: {len(jobs)} matched of {registered} registered; {n}/{len(data_D)} datums scored by the privileged teacher, mean token delta {tot / max(1, n):+.3f}", flush=True)
        return data_D, meta_D

    cb_train.assemble_training_data = patched


def main(cli: CLI) -> None:
    print("reward env:", {k: os.environ.get(k) for k in ("COMPILE_W", "STRICT_W", "MINIMAL_W", "STRUCTURE_W", "LENGTH_W", "THINK_BUDGET", "MAX_GEN_TOKENS", "COMPILE_SLOTS", "COMPILE_TIMEOUT", "REWARD_MODE", "SIM_W", "SIM_POW", "KEEP", "REFS", "GATE", "GATE_FLOOR", "DELETE_FREE", "DELETE_ZERO", "TEACHER_DEMOS", "RETRY_TURNS")}, flush=True)
    common.load_env()
    train_path = build_train_file(cli)
    val_path = str(common.DATA / cli.val) if cli.val else None
    from fix_env import FixDatasetBuilder
    builder = FixDatasetBuilder(train_data=str(train_path), test_data=val_path,
                                batch_size=cli.batch_size, group_size=cli.group_size, seed=cli.seed, skip=cli.skip_rows)
    cfg = train.Config(
        model_name=common.MODEL, recipe_name=f"latex_fix_{cli.mode}", renderer_name=common.RENDERER,
        log_path=cli.log_path, dataset_builder=builder, learning_rate=cli.lr, max_tokens=cli.max_tokens,
        eval_every=cli.eval_every, save_every=cli.save_every, load_checkpoint_path=cli.load_checkpoint_path,
        kl_penalty_coef=cli.kl_penalty_coef, lora_rank=cli.lora_rank, max_steps=cli.max_steps,
        wandb_project=cli.wandb_project, wandb_name=Path(cli.log_path).name if cli.wandb_project else None,
        async_config=train.AsyncConfig(max_steps_off_policy=cli.async_steps, groups_per_batch=cli.batch_size) if cli.async_steps else None,
    )
    if cli.kl_penalty_coef > 0:
        cfg = chz.replace(cfg, kl_reference_config=train.KLReferenceConfig(base_model=common.MODEL, load_checkpoint_path=cli.kl_ref))
    if cli.teacher_w > 0:
        install_teacher_hook(cli.teacher_w, cli.teacher_max)
    if cli.sdft_w > 0:
        os.environ["SDFT_W"] = str(cli.sdft_w)
        install_sdft_hook(cli.sdft_w, cli.sdft_ref)
    cli_utils.check_log_dir(cli.log_path, behavior_if_exists="resume")
    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main(chz.entrypoint(CLI, argv=sys.argv[1:]))
