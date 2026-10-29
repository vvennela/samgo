#!/bin/bash
# Train v1 (BrickGPT replication) then v2 (35-part palette) on Fireworks serverless, keeping the Mac awake.
# The account runs one Glimmer training at a time, so they go in sequence.
REPO=$(cd "$(dirname "$0")/.." && pwd)
COOKBOOK=$HOME/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0
PY=$HOME/.venvs/fwcookbook/bin/python
mkdir -p "$REPO/runs/train"
cd "$COOKBOOK" || exit 1
export PYTHONPATH="$REPO:$COOKBOOK"
caffeinate -dimsu bash -c "
  $PY -u $REPO/brickgpt/train_serverless.py --run-id brickgpt-v1 --palette brickgpt --batch 64 \
      --train data/sft/train_c2.jsonl --eval data/sft/eval_500.jsonl &&
  $PY -u $REPO/brickgpt/train_serverless.py --run-id samgo-v2 --palette v2 --batch 64 \
      --train data/v2/train_c2.jsonl --eval data/v2/test_c1.jsonl
"
