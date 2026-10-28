#!/bin/bash
# Pilot orchestrator, run as EC2 user-data on a 4-GPU box (g6e.12xlarge or g5.12xlarge).
# Baseline -> gate -> one expert-iteration round -> re-test -> report -> S3 -> shutdown.
# A dead-man timer stops the instance after MAX_MIN minutes no matter what.
set -uo pipefail
BUCKET=s3://samgo-runs-811269446285
RUN=${RUN:-pilot-001}
MAX_MIN=${MAX_MIN:-300}
MODEL=/opt/models/glimmer
W=/opt/run
OUT=$W/out

shutdown -h +"$MAX_MIN"
mkdir -p "$OUT" && cd "$W"
export HOME=/root PATH=/root/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONPATH=$W/code PYTHONUNBUFFERED=1

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/run.log"; }
phase() { echo "{\"phase\": \"$1\", \"t\": \"$(date -u +%FT%TZ)\"}" > "$OUT/phase.json"; log "phase: $1"; }
sync_out() { aws s3 sync "$OUT" "$BUCKET/$RUN/" --quiet --exclude "*/adapter/*" ; }
finish() {
  phase "${1:-done}"
  sync_out; aws s3 sync "$OUT" "$BUCKET/$RUN/" --quiet
  log "shutting down"; sync_out
  shutdown -h now
  exit 0
}

# Heartbeat: GPU stats and every log to S3 once a minute.
( while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
      | sed "s/^/$(date -u +%T) /" >> "$OUT/gpu.log" 2>/dev/null
    sync_out
    sleep 60
  done ) &

phase setup
aws s3 cp "$BUCKET/$RUN/code.tgz" /tmp/code.tgz && mkdir -p "$W/code" && tar xzf /tmp/code.tgz -C "$W/code" || finish failed_code_download
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

# Gate: don't pay for training if the task is broken (k=1 near 0%) or already solved (>90% overall).
GATE=$(python - <<'EOF'
import json
p = json.load(open("/opt/run/out/base_test/progress.json"))
k1 = p["pass_rate_by_k"].get("1", 0)
print("stop" if k1 < 0.02 or p["pass_rate"] > 0.90 else "go")
EOF
)
log "gate after baseline: $GATE"
if [ "$GATE" = "stop" ]; then python -m pilot.report "$OUT" >> "$OUT/run.log" 2>&1; finish done_gate_stop; fi

phase baseline_train
python -m pilot.sample --model "$MODEL" --tasks "$W/tasks/train.jsonl" --out "$OUT/base_train" --n 8 \
  > "$OUT/base_train.log" 2>&1 || finish failed_baseline_train

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
