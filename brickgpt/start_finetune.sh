#!/bin/bash
# Upload the BrickGPT replication data and start the free Glimmer LoRA fine-tune on Fireworks.
# Fireworks blocks AI agents from mutating commands, so a person runs this once:  ! bash brickgpt/start_finetune.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
F="firectl --api-key $FIREWORKS_API_KEY -a vvennela"
command -v firectl >/dev/null || F="/opt/homebrew/Cellar/firectl/64/bin/$F"
$F dataset create brickgpt-train-c2 data/sft/train_c2.jsonl      # 85,208 rows, 176 MB
$F dataset create brickgpt-eval-500 data/sft/eval_500.jsonl      # 500 held-out test-split rows
# BrickGPT: LoRA r32, 3 epochs on a 1B model; a 30B model on 2 captions/structure gets 1 epoch.
# --generation-eval saves base vs trained answers on the eval set, so we can verify builds without a deployment.
$F sftj create \
  --job-id brickgpt-glimmer-v1 --display-name "BrickGPT replication on Muse Glimmer 30B" \
  --base-model accounts/fireworks/models/muse-glimmer-30b \
  --dataset brickgpt-train-c2 --evaluation-dataset brickgpt-eval-500 \
  --output-model brickgpt-glimmer-v1 \
  --lora-rank 32 --epochs 1 --max-context-length 8192 --generation-eval
echo "started; check with: firectl -a vvennela sftj get brickgpt-glimmer-v1"
