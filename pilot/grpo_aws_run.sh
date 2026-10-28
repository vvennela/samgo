#!/bin/bash
# GRPO on a g6e.12xlarge from the pilot image (model + venv on disk), fetched and run by hold_userdata.sh.
# vLLM on GPUs 0,1 (TP2), the LoRA policy on GPUs 2,3. Budget: dead-man at MAX_MIN (~$28 at $10.49/hr).
set -uo pipefail
RUN=${RUN:-grpo-aws}
BUCKET=s3://samgo-runs-811269446285
LIVE=$BUCKET/live/$RUN
ARCHIVE=$BUCKET/runs/$RUN
MAX_MIN=${MAX_MIN:-160}
OUT=/opt/run/out/$RUN
shutdown -h +"$MAX_MIN"
mkdir -p "$OUT" /opt/grpo
phase() { echo "{\"phase\": \"$1\", \"t\": \"$(date -u +%FT%TZ)\"}" > "$OUT/phase.json"; aws s3 cp --quiet "$OUT/phase.json" "$LIVE/phase.json"; }
sync_live() {
  grep -h -E "Traceback|Error|error:|out of memory|OutOfMemory|Killed" "$OUT"/*.log 2>/dev/null | sort -u | tail -30 > "$OUT/errors.txt"
  grep -E "^\[grpo\]" "$OUT/grpo.log" 2>/dev/null | tail -40 > "$OUT/grpo_lines.txt"
  aws s3 sync "$OUT" "$LIVE/" --quiet --exclude "*" --include "phase.json" --include "rounds.jsonl" --include "probe_*.json" \
    --include "errors.txt" --include "grpo_lines.txt" --include "gpu.log"
}
( while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed "s/^/$(date -u +%T) /" >> "$OUT/gpu.log"
    sync_live; sleep 60
  done ) &
phase setup
aws s3 cp --quiet "$BUCKET/code/grpo-code.tgz" /tmp/code.tgz && tar xzf /tmp/code.tgz -C /opt/grpo || { phase failed_code; sync_live; shutdown -h now; exit 1; }
source /opt/venv/bin/activate
export HOME=/root PYTHONPATH=/opt/grpo PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
phase running
python -m pilot.grpo --model /opt/models/glimmer --tasks /opt/grpo/tasks/train.jsonl --out "$OUT" \
  --roll-gpus 0,1 --train-gpus 2,3 --probe /opt/grpo/tasks/probe.jsonl --max-minutes $((MAX_MIN - 15)) \
  --per-level 8 --group 8 --minibatches 4 --lr 5e-5 > "$OUT/grpo.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then phase done; else phase "failed_rc$rc"; fi
sync_live
# Archive everything but intermediate adapters (keep the last one).
LAST=$(ls -d "$OUT"/round*/ 2>/dev/null | sort -V | tail -1)
aws s3 sync "$OUT" "$ARCHIVE/" --quiet --exclude "round*/*" && [ -n "$LAST" ] && aws s3 sync "$LAST" "$ARCHIVE/$(basename "$LAST")/" --quiet
shutdown -h now
