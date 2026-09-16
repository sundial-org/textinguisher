"""LoRA SFT on (fix prompt -> edit blocks) pairs via the Tinker API.

Usage: train_sft.py [--data ../data/train.jsonl] [--epochs 3] [--batch-size 64] [--lr 1e-4] [--lora-rank 32]
"""
import argparse
import json
import random
import time

import common

common.load_env()
import tinker  # noqa: E402
from tml_renderers import chat  # noqa: E402
from tml_renderers.tinker import training_example_to_tinker_model_input_and_weights  # noqa: E402


def example_to_datum(ex: dict, renderer) -> tinker.Datum:
    if common.RENDERER != "tml_v0":  # cookbook renderer: supervised example = (model_input, per-token weights)
        model_input, w = renderer.build_supervised_examples(common.cookbook_messages(ex["prompt"], ex["target"]))[-1]
        chunks = list(model_input.chunks)
        last = chunks[-1]
        flat = []
        for c in chunks:
            flat += list(c.tokens) if hasattr(c, "tokens") else [0] * c.length
        w = [float(x) for x in w.tolist()]
        trimmed = tinker.ModelInput(chunks=chunks[:-1] + [type(last)(tokens=last.tokens[:-1])])
        return tinker.Datum(model_input=trimmed, loss_fn_inputs={"target_tokens": flat[1:], "weights": w[1:]})
    msgs = [
        chat.Message(content=chat.Text(common.SYSTEM), author=chat.Author(chat.AuthorKind.System)),
        chat.Message(content=chat.Text(ex["prompt"]), author=chat.Author(chat.AuthorKind.User)),
        chat.Message(content=chat.Text(ex["target"]), author=chat.Author(chat.AuthorKind.Model)),
    ]
    rendered = renderer.render_for_sft(msgs)
    assert len(rendered) == 1
    model_input, w = training_example_to_tinker_model_input_and_weights(rendered[0])
    chunks = list(model_input.chunks)
    last = chunks[-1]
    assert hasattr(last, "tokens") and len(last.tokens) > 1
    flat = []
    for c in chunks:
        flat += list(c.tokens) if hasattr(c, "tokens") else [0] * c.length
    trimmed = tinker.ModelInput(chunks=chunks[:-1] + [type(last)(tokens=last.tokens[:-1])])
    return tinker.Datum(
        model_input=trimmed,
        loss_fn_inputs={"target_tokens": flat[1:], "weights": [float(x) for x in w[1:]]},
    )


def with_retries(fn, what: str, attempts: int = 6):
    for a in range(attempts):
        try:
            return fn()
        except Exception as e:
            if a == attempts - 1:
                raise
            wait = min(30 * 2 ** a, 600)
            print(f"{what} failed ({e}); retry {a + 1}/{attempts} in {wait}s")
            time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.DATA / "train.jsonl"))
    ap.add_argument("--val-data", default=str(common.DATA / "val.jsonl"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="cap train/val rows (smoke runs)")
    ap.add_argument("--load-state", default=None, help="tinker://.../weights/... state to continue from (rank must match)")
    args = ap.parse_args()

    renderer = common.get_renderer() if common.RENDERER == "tml_v0" else common.cookbook_renderer()
    train = [json.loads(l) for l in open(args.data)]
    val = [json.loads(l) for l in open(args.val_data)]
    if args.limit:
        train, val = train[: args.limit], val[: max(4, args.limit // 8)]
    print(f"train={len(train)} val={len(val)} base={common.MODEL}")

    train_datums = [example_to_datum(e, renderer) for e in train]
    val_datums = [example_to_datum(e, renderer) for e in val]

    tc = tinker.ServiceClient().create_lora_training_client(
        base_model=common.MODEL, rank=args.lora_rank)
    if args.load_state:
        tc.load_state(args.load_state).result()
        print(f"loaded state {args.load_state}")

    rng = random.Random(3)
    step, total = 0, args.epochs * max(1, len(train_datums) // args.batch_size)
    log = (common.PROJ / "out" / "train_log.jsonl").open("w")
    for epoch in range(args.epochs):
        rng.shuffle(train_datums)
        for i in range(0, len(train_datums) - args.batch_size + 1, args.batch_size):
            batch = train_datums[i:i + args.batch_size]
            lr = args.lr * max(0.1, 1.0 - step / total)

            def do_step():
                fb = tc.forward_backward(batch, loss_fn="cross_entropy")
                opt = tc.optim_step(tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
                return fb.result(), opt.result()

            fb_res, _ = with_retries(do_step, f"step {step}")
            lp = [x["logprobs"].to_torch() for x in fb_res.loss_fn_outputs]
            ws = [d.loss_fn_inputs["weights"].to_torch() for d in batch]
            nll = sum(-(l * w).sum().item() for l, w in zip(lp, ws)) / max(
                sum(w.sum().item() for w in ws), 1)
            print(f"epoch {epoch} step {step}/{total} lr={lr:.2e} nll={nll:.4f}")
            log.write(json.dumps({"epoch": epoch, "step": step, "lr": lr, "nll": nll}) + "\n")
            log.flush()
            step += 1

        fwd = with_retries(lambda: tc.forward(val_datums, loss_fn="cross_entropy").result(), "val forward")
        lp = [x["logprobs"].to_torch() for x in fwd.loss_fn_outputs]
        ws = [d.loss_fn_inputs["weights"].to_torch() for d in val_datums]
        vnll = sum(-(l * w).sum().item() for l, w in zip(lp, ws)) / sum(w.sum().item() for w in ws)
        print(f"epoch {epoch} VAL nll={vnll:.4f}")
        log.write(json.dumps({"epoch": epoch, "val_nll": vnll}) + "\n")
        log.flush()
        path = with_retries(
            lambda: tc.save_weights_for_sampler(name=f"epoch{epoch}").result(), "save").path
        print(f"sampler checkpoint: {path}")
        state = with_retries(lambda: tc.save_state(name=f"state{epoch}").result(), "save_state").path
        print(f"state checkpoint: {state}")  # RL (cookbook load_checkpoint_path) needs training weights, not sampler weights
        (common.PROJ / "out" / "checkpoints.txt").open("a").write(f"epoch{epoch}\t{path}\t{state}\n")

    log.close()


if __name__ == "__main__":
    main()
