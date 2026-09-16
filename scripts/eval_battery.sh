#!/bin/zsh
# Standard eval battery for one arm: real held-out TeX.SE (287), organic skew (184, compile-only),
# v1 (150), hard-fair (109), project (85).  usage: eval_battery.sh <arm> <tag> [sets...]
set -u
cd "$(dirname "$0")/.."
ARM=$1; TAG=$2; shift 2
[[ $# -eq 0 ]] && set -- texse_heldout skew_eval eval hard_eval3 project_eval
for S in "$@"; do
  LIM=(); [[ $S == eval ]] && LIM=(--limit 150)
  echo "=== $TAG $S ==="
  ../.venv/bin/python -u scripts/run_eval.py --data data/$S.jsonl --arm "$ARM" $LIM --out out/${S}_${TAG}.json 2>&1 \
    | grep -E "^fixed|^exact|^edits|^out tokens|^wall|Traceback|Error" | grep -v Warn
done
echo "=== $TAG done ==="
