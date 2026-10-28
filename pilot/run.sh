#!/bin/bash
# Pilot orchestrator, run as EC2 user-data on a 4-GPU box (g6e.12xlarge or g5.12xlarge).
# Baseline -> gate -> one expert-iteration round -> re-test -> report -> S3 -> shutdown.
# A dead-man timer stops the instance after MAX_MIN minutes no matter what.
#
# S3 layout: code/<run>.tgz (input), live/<run>/ (small monitoring files, wiped at start),
# runs/<run>/ (everything, written only when the run ends with a result). A crash writes
# only live/<run>/failure.txt, never runs/<run>/.
set -uo pipefail
BUCKET=s3://samgo-runs-811269446285
RUN=${RUN:-pilot-002}
MAX_MIN=${MAX_MIN:-300}
MODEL=/opt/models/glimmer
W=/opt/run
OUT=$W/out
LIVE=$BUCKET/live/$RUN
ARCHIVE=$BUCKET/runs/$RUN

shutdown -h +"$MAX_MIN"
mkdir -p "$OUT" && cd "$W"
export HOME=/root PATH=/root/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONPATH=$W/code PYTHONUNBUFFERED=1
aws s3 rm "$LIVE/" --recursive --quiet

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/run.log"; }
phase() { echo "{\"phase\": \"$1\", \"t\": \"$(date -u +%FT%TZ)\"}" > "$OUT/phase.json"; log "phase: $1"; }
sync_live() {
  # Only small monitoring files; error lines are collected so the full logs never need to leave the box.
  grep -h -E "Traceback|Error|error:|out of memory|OutOfMemory|Killed" "$OUT"/*.log 2>/dev/null | sort -u | tail -40 > "$OUT/errors.txt"
  aws s3 sync "$OUT" "$LIVE/" --quiet --exclude "*" --include "phase.json" --include "run.log" --include "gpu.log" \
    --include "errors.txt" --include "failure.txt" --include "plan.json" --include "*/progress.json" \
    --include "round1/train_log.jsonl" --include "round1/train_meta.json"
}
finish() {
  status=${1:-done}
  phase "$status"
  case "$status" in
    done*) aws s3 sync "$OUT" "$ARCHIVE/" --quiet && log "archived to $ARCHIVE" ;;
    *) newest=$(ls -t "$OUT"/*.log 2>/dev/null | grep -v -E "run.log|gpu.log" | head -1)
       { echo "status: $status"; echo "log: $newest"; tail -40 "$newest" 2>/dev/null; } > "$OUT/failure.txt" ;;
  esac
  log "shutting down"; sync_live
  shutdown -h now
  exit 0
}

# Heartbeat: GPU stats and monitoring files to S3 once a minute.
( while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
      | sed "s/^/$(date -u +%T) /" >> "$OUT/gpu.log" 2>/dev/null
    sync_live
    sleep 60
  done ) &

phase setup
aws s3 cp "$BUCKET/code/$RUN.tgz" /tmp/code.tgz && mkdir -p "$W/code" && tar xzf /tmp/code.tgz -C "$W/code" || finish failed_code_download
cp -r "$W/code/runs/pilot/tasks" "$W/tasks"
curl -LsSf https://astral.sh/uv/install.sh | sh >> "$OUT/setup.log" 2>&1
uv venv /opt/venv --python 3.12 >> "$OUT/setup.log" 2>&1
source /opt/venv/bin/activate
uv pip install vllm transformers peft accelerate hf_transfer "huggingface_hub[cli]" numpy scipy >> "$OUT/setup.log" 2>&1 \
  || finish failed_install
python -c "import vllm, transformers, peft, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__)" \
  | tee -a "$OUT/run.log"

phase download
hf download meta-models/Muse-Glimmer-30B --local-dir "$MODEL" >> "$OUT/setup.log" 2>&1 || finish failed_download
du -sh "$MODEL" | tee -a "$OUT/run.log"

phase baseline_test
python -m pilot.sample --model "$MODEL" --tasks "$W/tasks/test.jsonl" --out "$OUT/base_test" --n 8 \
  > "$OUT/base_test.log" 2>&1 || finish failed_baseline_test

# Gate + plan: train only on levels in the 10-90% band; hold out the next harder level.
python -m pilot.band "$OUT" >> "$OUT/run.log" 2>&1 || finish failed_plan
DECISION=$(python -c "import json; print(json.load(open('$OUT/plan.json'))['decision'])")
TRAIN_LEVELS=$(python -c "import json; print(','.join(json.load(open('$OUT/plan.json')).get('train_levels', [])))")
log "plan: $DECISION, train levels: $TRAIN_LEVELS"
if [ "$DECISION" = "stop" ]; then python -m pilot.report "$OUT" >> "$OUT/run.log" 2>&1; finish done_gate_stop; fi

phase baseline_train
python -m pilot.sample --model "$MODEL" --tasks "$W/tasks/train.jsonl" --out "$OUT/base_train" --n 8 \
  --levels "$TRAIN_LEVELS" > "$OUT/base_train.log" 2>&1 || finish failed_baseline_train

phase train
python -m pilot.train --model "$MODEL" --samples "$OUT/base_train/samples.jsonl" --out "$OUT/round1" \
  --cap 1 --max-examples ${MAX_EX:-400} > "$OUT/train.log" 2>&1 || finish failed_train

phase round1_test
if ! python -m pilot.sample --model "$MODEL" --tasks "$W/tasks/test.jsonl" --out "$OUT/r1_test" --n 8 \
     --lora "$OUT/round1/adapter" > "$OUT/r1_test.log" 2>&1; then
  log "LoRA serving failed; merging adapter into base weights"
  phase merge
  CUDA_VISIBLE_DEVICES= python -m pilot.merge "$MODEL" "$OUT/round1/adapter" /opt/models/merged > "$OUT/merge.log" 2>&1 || finish failed_merge
  phase round1_test_merged
  python -m pilot.sample --model /opt/models/merged --tasks "$W/tasks/test.jsonl" --out "$OUT/r1_test" --n 8 \
    > "$OUT/r1_test.log" 2>&1 || finish failed_round1_test
fi

phase report
python -m pilot.report "$OUT" >> "$OUT/run.log" 2>&1
finish done
