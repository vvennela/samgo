#!/bin/bash
# Shortened phase 2 for a run whose baseline was cut after its first chunk (slow hardware).
# Runs on the instance after the original orchestrator is stopped; keeps its heartbeat and dead-man.
# Test set = the tasks the partial baseline covered; everything else as in run.sh, smaller.
set -uo pipefail
RUN=${RUN:-pilot-002}
BUCKET=s3://samgo-runs-811269446285
LIVE=$BUCKET/live/$RUN
ARCHIVE=$BUCKET/runs/$RUN
W=/opt/run
OUT=$W/out
MODEL=/opt/models/glimmer
source /opt/venv/bin/activate
export HOME=/root PYTHONPATH=$W/code PYTHONUNBUFFERED=1
shutdown -h +"${MAX_MIN:-150}"   # own dead-man: after a reboot run.sh's timer is gone

# Own heartbeat (run.sh's is gone after a reboot): GPU stats + monitoring files once a minute.
( while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
      | sed "s/^/$(date -u +%T) /" >> "$OUT/gpu.log" 2>/dev/null
    grep -h -E "Traceback|Error|error:|out of memory|OutOfMemory|Killed" "$OUT"/*.log 2>/dev/null | sort -u | tail -40 > "$OUT/errors.txt"
    aws s3 sync "$OUT" "s3://samgo-runs-811269446285/live/${RUN:-pilot-002}/" --quiet --exclude "*" --include "phase.json" \
      --include "run.log" --include "gpu.log" --include "errors.txt" --include "plan.json" --include "*/progress.json" \
      --include "round1/train_log.jsonl" --include "round1/train_meta.json"
    sleep 60
  done ) &

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/run.log"; }
phase() { echo "{\"phase\": \"$1\", \"t\": \"$(date -u +%FT%TZ)\"}" > "$OUT/phase.json"; log "phase: $1"; }
finish() {
  status=${1:-done}
  phase "$status"
  case "$status" in
    done*) aws s3 sync "$OUT" "$ARCHIVE/" --quiet && log "archived to $ARCHIVE" ;;
    *) newest=$(ls -t "$OUT"/*.log 2>/dev/null | grep -v -E "run.log|gpu.log" | head -1)
       { echo "status: $status"; echo "log: $newest"; tail -40 "$newest" 2>/dev/null; } > "$OUT/failure.txt"
       # Keep the real baseline data even if a later step fails.
       aws s3 cp "$OUT/base_test/samples.jsonl" "$ARCHIVE/base_test/samples.jsonl" --quiet ;;
  esac
  aws s3 sync "$OUT" "$LIVE/" --quiet --exclude "*" --include "phase.json" --include "run.log" --include "failure.txt" \
    --include "plan.json" --include "*/progress.json" --include "round1/train_*"
  shutdown -h now
  exit 0
}

log "phase 2 takeover: baseline cut after $(wc -l < "$OUT/base_test/samples.jsonl") samples"
python - <<'EOF' || finish failed_subset
import json
ids = {json.loads(l)["id"] for l in open("/opt/run/out/base_test/samples.jsonl")}
rows = [l for l in open("/opt/run/tasks/test.jsonl") if json.loads(l)["id"] in ids]
open("/opt/run/tasks/test_sub.jsonl", "w").writelines(rows)
print(f"test subset: {len(rows)} tasks")
EOF

phase plan
python -m pilot.band "$OUT" >> "$OUT/run.log" 2>&1 || finish failed_plan
DECISION=$(python -c "import json; print(json.load(open('$OUT/plan.json'))['decision'])")
TRAIN_LEVELS=$(python -c "import json; print(','.join(json.load(open('$OUT/plan.json')).get('train_levels', [])))")
log "plan: $DECISION, train levels: $TRAIN_LEVELS"
if [ "$DECISION" = "stop" ]; then python -m pilot.report "$OUT" >> "$OUT/run.log" 2>&1; finish done_gate_stop; fi

phase baseline_train
python -m pilot.sample --model "$MODEL" --tasks "$W/tasks/train.jsonl" --out "$OUT/base_train" --n 4 \
  --levels "$TRAIN_LEVELS" > "$OUT/base_train.log" 2>&1 || finish failed_baseline_train

phase train
python -m pilot.train --model "$MODEL" --samples "$OUT/base_train/samples.jsonl" --out "$OUT/round1" \
  --cap 1 --max-examples 400 > "$OUT/train.log" 2>&1 || finish failed_train

phase round1_test
if ! python -m pilot.sample --model "$MODEL" --tasks "$W/tasks/test_sub.jsonl" --out "$OUT/r1_test" --n 4 \
     --lora "$OUT/round1/adapter" > "$OUT/r1_test.log" 2>&1; then
  log "LoRA serving failed; merging adapter into base weights"
  phase merge
  CUDA_VISIBLE_DEVICES= python -m pilot.merge "$MODEL" "$OUT/round1/adapter" /opt/models/merged > "$OUT/merge.log" 2>&1 || finish failed_merge
  phase round1_test_merged
  python -m pilot.sample --model /opt/models/merged --tasks "$W/tasks/test_sub.jsonl" --out "$OUT/r1_test" --n 4 \
    > "$OUT/r1_test.log" 2>&1 || finish failed_round1_test
fi

phase report
python -m pilot.report "$OUT" >> "$OUT/run.log" 2>&1
finish done
