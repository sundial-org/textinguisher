"""Single-shot RL env: the prod fix prompt -> edit blocks, rewarded by the compiler (cookbook MessageEnv).

Reward: COMPILE_W if the patched project compiles, +STRICT_W if strict (PDF text/pages match refs.json), -0.2 if
the edits do not parse/apply, all scaled by a keep-factor that decays linearly with content removed (non-comment
chars deleted or commented out): full up to DELETE_FREE chars, zero at DELETE_ZERO. Group reward: among the compiling
samples of a GRPO group the smallest edit gets +MINIMAL_W, the largest 0 (reference-free "prefer the minimal fix").
Run 1 (1.0/0.5, soft -0.6 deletion cap) learned to inline macros and delete until the doc compiled; run 2 (1.0/0.5,
hard cap) fixed the deletion but still preferred "a compiling fix" over "the fix"; run 3 = 0.5/1.0 + minimality (worse).
STRUCTURE_W: reference-free intent check -- every figure, environment, cite/ref/label, item, macro definition and package
present in the broken source must survive in the fixed source; each one lost costs STRUCTURE_W of the compile credit.
THINK_BUDGET > 0 lets the policy reason before the edits (system-prompt instruction + larger generation budget).
Pool rows get pseudo-refs (data/refs_pool.json: PDF of the compile-verified teacher fix) so strict applies to them too;
an LLM judge for faithfulness was tried and rejected (Haiku/Gemini mislabel 4/7 ground-truth fixes as hacks).
"""
import asyncio
import json
import os
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import chz
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder

import agent_env
import common
from run_eval import score_response

DELETE_FREE, DELETE_ZERO = int(os.environ.get("DELETE_FREE", 25)), int(os.environ.get("DELETE_ZERO", 150))  # metric-matched: 40, 41
COMPILE_W, STRICT_W, MINIMAL_W, STRUCTURE_W = (float(os.environ.get(k, d)) for k, d in (
    ("COMPILE_W", 1.0), ("STRICT_W", 0.5), ("MINIMAL_W", 0.0), ("STRUCTURE_W", 0.0)))
LENGTH_W = float(os.environ.get("LENGTH_W", "0"))  # reward cost of a 1,000-char thought: makes thinking adaptive
# REWARD_MODE=sim: the main term is a continuous PDF agreement with the reference (text + page rasters, common.pdf_score)
# instead of the binary strict bonus and the deletion keep-factor: COMPILE_W + SIM_W * sim ** SIM_POW (KEEP=1 re-enables the cap).
# REWARD_MODE=gate: COMPILE_W only when the compiled PDF matches the reference (sim >= GATE), else GATE_FLOOR if it merely compiles,
# so deleting content earns nothing more than failing; no deletion heuristic needed.
REWARD_MODE = os.environ.get("REWARD_MODE", "strict")
KEPT_MAX = int(os.environ.get("KEPT_MAX", 40))
RETRY_TURNS = int(os.environ.get("RETRY_TURNS", 0))  # >0: a failed compile returns the new error state as the next user turn (the product's loop)  # REWARD_MODE=ke: the reporting threshold for "content kept"
GATE, GATE_FLOOR = float(os.environ.get("GATE", "0.95")), float(os.environ.get("GATE_FLOOR", "0.0"))
SIM_W, SIM_POW, KEEP = float(os.environ.get("SIM_W", "1.0")), float(os.environ.get("SIM_POW", "2.0")), os.environ.get("KEEP", "0") == "1"
THINK_BUDGET = common.THINK_BUDGET
# TEACHER_DEMOS=data/train_astra.jsonl (comma list of prompt/target/id files): when no sample in a group matches the reference and a
# verified teacher fix exists for the prompt, the fix is queued; train_rl adds it to the step as a cross-entropy datum weighted TEACHER_W
# (interleaved teacher correction, no separate SFT phase). Own exact samples are not queued: they carry no new knowledge.
TEACHER_DEMOS: dict[str, str] = {}
for _f in filter(None, os.environ.get("TEACHER_DEMOS", "").split(",")):
    for _l in Path(_f if _f.startswith("/") else str(common.DATA / Path(_f).name)).read_text().splitlines():
        if _l.strip():
            _r = json.loads(_l); TEACHER_DEMOS.setdefault(_r["id"], _r["target"])
TEACHER_QUEUE: list[dict] = []
SDFT_ROWS: dict[tuple, tuple[str, str]] = {}  # tuple(action tokens) -> (prompt, verified fix) for train_rl's demonstration-conditioned scoring
STRUCT_RE = re.compile(r"\\(includegraphics|begin\{[a-zA-Z*]+\}|cite[a-z]*|ref|label|item|section\*?|subsection\*?|caption|"
                       r"newcommand|renewcommand|def|newenvironment|usepackage|addplot|draw|node|footnote)(?![a-zA-Z])")
COMMENT_RE = re.compile(r"(?<!\\)%.*")


def _content(s: str) -> int:
    return len(re.sub(r"\s+", "", COMMENT_RE.sub("", s)))


def deleted_chars(response: str) -> int:
    edits = common.parse_file_edits(response) if "<edit file=" in response else [
        ("main.tex", s, r) for s, r in common.parse_edits(response)]
    return sum(max(0, _content(s) - _content(r)) for _, s, r in edits)


def edit_size(response: str) -> int:
    """Content chars touched by the edits (search + replace), the minimality signal."""
    edits = common.parse_file_edits(response) if "<edit file=" in response else [
        ("main.tex", s, r) for s, r in common.parse_edits(response)]
    return sum(_content(s) + _content(r) for _, s, r in edits)


def structure_lost(broken: str, fixed: str) -> int:
    """Number of structural constructs (figures, environments, cites, items, macro definitions...) the fix removed."""
    from collections import Counter
    cb, cf = Counter(STRUCT_RE.findall(COMMENT_RE.sub("", broken))), Counter(STRUCT_RE.findall(COMMENT_RE.sub("", fixed)))
    return sum(max(0, cb[k] - cf[k]) for k in cb)


def reward_of(res: dict, response: str, lost: int = 0, extra_deleted: int = 0) -> float:
    if not res["applied"]:
        return -0.2
    if not res["compiled"]:
        return 0.0
    deleted = deleted_chars(response) + extra_deleted  # extra = content removed by earlier turns of a retry episode
    keep = 1.0 - min(1.0, max(0, deleted - DELETE_FREE) / (DELETE_ZERO - DELETE_FREE))
    keep *= max(0.0, 1.0 - STRUCTURE_W * lost)
    if REWARD_MODE == "ke":  # metric-matched: 1 if compiles with <= KEPT_MAX chars deleted, +1 if the PDF matches the reference
        return float(deleted <= KEPT_MAX) + float(res["strict"])
    if REWARD_MODE == "gate" and res.get("sim") is not None:
        return COMPILE_W if res["sim"] >= GATE else GATE_FLOOR
    if REWARD_MODE == "sim" and res.get("sim") is not None:
        return (COMPILE_W + SIM_W * res["sim"] ** SIM_POW) * (keep if KEEP else 1.0)
    return (COMPILE_W + (STRICT_W if res["strict"] else 0.0)) * keep


@dataclass
class FixEnv(MessageEnv):
    row: dict
    ref: dict | None = None
    outcome: tuple[bool, int] | None = None  # (compiled, edit_size) for the group reward
    attempt: int = 0
    doc: str | None = None  # current document text after earlier turns (single-file rows)
    deleted_before: int = 0

    async def initial_observation(self) -> list:
        return [{"role": "system", "content": common.system_prompt()}, {"role": "user", "content": self.row["prompt"]}]

    async def step(self, message) -> MessageStepResult:
        text = agent_env._text(message)
        think = re.search(r"<think>(.*?)</think>", text, re.S)
        text = text[think.end():] if think else text
        async with agent_env.compile_slot():
            res = await asyncio.to_thread(score_response, self.row, text, self.ref, self.doc)
        if (RETRY_TURNS and self.attempt < RETRY_TURNS and not res["compiled"] and res.get("retry_doc")
                and self.row.get("lane") != "project"):  # compiler feedback turn: same episode, new error state
            self.attempt += 1
            self.doc = res["retry_doc"]
            self.deleted_before += deleted_chars(text)
            prompt = common.build_fix_prompt("main.tex", res["retry_error_lines"], res["retry_log_tail"], res["retry_doc"],
                                             full_lines=500, max_chars=16000)
            return MessageStepResult(reward=-0.2 if not res["applied"] else 0.0, episode_done=False,
                                     next_messages=[{"role": "user", "content": prompt}], metrics={"retry_turn": 1.0})
        lost = 0
        if STRUCTURE_W and res["applied"] and self.row.get("lane") != "project":
            from run_eval import broken_text
            broken = broken_text(self.row)
            lost = structure_lost(broken, common.apply_edits(broken, common.parse_edits(text))[0])
        rew = reward_of(res, text, lost, self.deleted_before) - (LENGTH_W * len(think.group(1)) / 1000 if think else 0.0)
        self.outcome = (bool(res["compiled"]), edit_size(text), bool(res["strict"]))
        metrics = {"compiled": float(res["compiled"]), "strict": float(res["strict"]), "turns": float(self.attempt + 1), "sim": float(res.get("sim") or 0),
                   "text_sim": float(res.get("text_sim") or 0), "img_sim": float(res.get("img_sim") or 0),
                   "applied": float(res["applied"]), "deleted_chars": float(deleted_chars(text) + self.deleted_before),
                   "diff_lines": float(res["diff_lines"] or 0), "structure_lost": float(lost),
                   "think_chars": float(len(think.group(1))) if think else 0.0}
        return MessageStepResult(reward=rew, episode_done=True, next_messages=[], metrics=metrics)


@dataclass
class FixGroupBuilder(EnvGroupBuilder):
    row: dict
    ref: dict | None
    num_envs: int
    max_generation_tokens: int = int(os.environ.get("MAX_GEN_TOKENS", 1500)) + THINK_BUDGET  # base model thinks natively: raise
    max_trajectory_tokens: int = 16000 + THINK_BUDGET  # multi-file prompts with context run to ~10k tokens
    envs: list = field(default_factory=list)

    async def make_envs(self) -> Sequence[Env]:
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.rl.message_env import EnvFromMessageEnv
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        renderer = get_renderer(common.RENDERER, get_tokenizer(common.MODEL))
        self.envs = [FixEnv(self.row, self.ref) for _ in range(self.num_envs)]
        return [EnvFromMessageEnv(renderer, e, failed_parse_reward=-0.2, max_trajectory_tokens=self.max_trajectory_tokens,
                                  max_generation_tokens=self.max_generation_tokens) for e in self.envs]

    async def compute_group_rewards(self, trajectory_group, env_group) -> list[tuple[float, dict]]:
        if TEACHER_DEMOS and self.row["id"] in TEACHER_DEMOS and not any(e.outcome and e.outcome[2] for e in self.envs):
            TEACHER_QUEUE.append({"id": self.row["id"], "prompt": self.row["prompt"], "target": TEACHER_DEMOS[self.row["id"]]})
        if TEACHER_DEMOS and self.row["id"] in TEACHER_DEMOS and os.environ.get("SDFT_W"):
            for t in trajectory_group:  # keyed by the sampled tokens: trajectory objects may be copied before training
                if t.transitions:
                    SDFT_ROWS[tuple(int(x) for x in t.transitions[0].ac.tokens)] = (self.row["prompt"], TEACHER_DEMOS[self.row["id"]])
        ok = sorted({e.outcome[1] for e in self.envs if e.outcome and e.outcome[0]})
        out = []
        for e in self.envs:
            if not (e.outcome and e.outcome[0]):
                out.append((0.0, {}))
                continue
            rank = ok.index(e.outcome[1]) / max(1, len(ok) - 1) if len(ok) > 1 else 0.5
            out.append((MINIMAL_W * (1.0 - rank), {"minimal_bonus": MINIMAL_W * (1.0 - rank)}))
        return out

    def logging_tags(self) -> list[str]:
        return ["latex_fix", self.row.get("lane") or "single", self.row["mutation"]]


class FixDataset(RLDataset):
    def __init__(self, data: str | Path, batch_size: int, group_size: int, seed: int = 0, skip: int = 0):
        self.rows = [json.loads(l) for l in Path(data).read_text().splitlines() if l.strip()]
        random.Random(seed).shuffle(self.rows)
        self.rows = self.rows[skip:]  # continue a previous pass: skip the prompts it already visited
        self.refs = common.load_refs()
        self.batch_size, self.group_size = batch_size, group_size

    def get_batch(self, index: int) -> Sequence[FixGroupBuilder]:
        return [FixGroupBuilder(row, self.refs.get(row["doc"]), self.group_size)
                for row in self.rows[index * self.batch_size:(index + 1) * self.batch_size]]

    def __len__(self) -> int:
        return len(self.rows) // self.batch_size


@chz.chz
class FixDatasetBuilder(RLDatasetBuilder):
    train_data: str
    test_data: str | None = None
    batch_size: int = 16
    group_size: int = 8
    seed: int = 0
    skip: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        return (FixDataset(self.train_data, self.batch_size, self.group_size, self.seed, self.skip),
                FixDataset(self.test_data, 16, 1) if self.test_data else None)


async def _selftest() -> None:
    """Score the ground-truth target (reward 1.5), a deletion hack and garbage on one labeled row."""
    ds = FixDataset(common.DATA / "texse_train.jsonl", 1, 1)
    b = ds.get_batch(0)[0]
    for name, text in [("target", b.row["target"]), ("garbage", "no edits here"),
                       ("delete", common.format_edits([(e["replace"], "") for e in b.row["break_edits"]]))]:
        r = await FixEnv(b.row, b.ref).step({"role": "assistant", "content": text})
        print(f"{name:8s} reward={r.reward:+.2f} {r.metrics}")


if __name__ == "__main__":
    asyncio.run(_selftest())
