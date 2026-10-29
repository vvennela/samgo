#!/bin/bash
# Upload the v2 (35-part palette) data and start the second free Glimmer LoRA fine-tune.
# Fireworks blocks AI agents from mutating commands, so a person runs this once:  ! bash brickgpt/start_finetune_v2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
F="firectl --api-key $FIREWORKS_API_KEY -a vvennela"
command -v firectl >/dev/null || F="/opt/homebrew/Cellar/firectl/64/bin/$F"
$F dataset create samgo-v2-train data/v2/train_c2.jsonl
$F dataset create samgo-v2-eval data/v2/test_c1.jsonl
# v2 rows average 1.9k tokens (max 6.5k over a 400-row sample), so 8k fits every row.
$F sftj create \
  --job-id samgo-glimmer-v2 --display-name "SAMgo 35-part palette on Muse Glimmer 30B" \
  --base-model accounts/fireworks/models/muse-glimmer-30b \
  --dataset samgo-v2-train --evaluation-dataset samgo-v2-eval \
  --output-model samgo-glimmer-v2 \
  --lora-rank 32 --epochs 1 --max-context-length 8192 --generation-eval
echo "started; check with: firectl -a vvennela sftj get samgo-glimmer-v2"
