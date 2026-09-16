"""SFT rows from compile-verified, deletion-bounded (<=40 chars removed) teacher fixes on the multi-file training projects (context prompts), plus a replay of
single-file rows so the model keeps its single-file behaviour.  Writes data/multi_sft.jsonl.
  build_multi_sft.py [<set>=<teacher result json> ...]   (default: the two SRC entries below)"""
import json
import random
import sys

import common
from fix_env import deleted_chars

SRC = [tuple(a.split("=", 1)) for a in sys.argv[1:]] or [("multi2_train", "out/multi2_train_claude-fable-5.1.json"), ("skew_train", "out/skew_train_gemini-3.7-flash.json")]
REPLAY = 600


def main() -> None:
    out, stats = [], {}
    for name, res in SRC:
        rows = {r["id"]: r for r in (json.loads(l) for l in (common.DATA / f"{name}.jsonl").read_text().splitlines() if l.strip())}
        results = json.load(open(common.PROJ / res))["results"]
        keep = [x for x in results if x.get("compiled") and x.get("applied") and x["id"] in rows
                and deleted_chars(common.THINK_RE.sub("", x["response"])) <= 40]  # no deletion-shaped teacher fixes
        strict = [x for x in keep if x.get("strict")]
        stats[name] = (len(results), len(keep), len(strict))
        for x in keep:
            target = common.THINK_RE.sub("", x["response"]).strip()
            out.append({"id": x["id"], "prompt": rows[x["id"]]["prompt"], "target": target, "source": f"teacher:{name}"})
    single = [json.loads(l) for l in (common.DATA / "trainA.jsonl").read_text().splitlines() if l.strip()]
    rng = random.Random(0)
    out += [{**r, "source": "replay"} for r in rng.sample(single, min(REPLAY, len(single)))]
    rng.shuffle(out)
    (common.DATA / "multi_sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"teacher rows (n, compiled, strict): {stats}; replay {REPLAY}; wrote {len(out)} -> data/multi_sft.jsonl")


if __name__ == "__main__":
    main()
