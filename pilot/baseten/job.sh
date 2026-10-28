#!/bin/bash
# Pilot phase 2 on one 2xH100 node (Baseten or Modal): same-hardware baseline, round-1 training, re-test, report.
# Every step's output is copied to $SAVE (Baseten checkpoint dir or a Modal volume) as soon as it exists.
# SKIP_SETUP=1 when the image already has the packages and the model is already at $MODEL.
set -uo pipefail
CODE=$(pwd)
OUT=/opt/run/out
MODEL=${MODEL:-${BT_PROJECT_CACHE_DIR:-/root/.cache/user_artifacts}/glimmer}
SAVE=${SAVE:-${BT_CHECKPOINT_DIR:-/tmp/save}/results}
mkdir -p "$OUT" "$SAVE" /opt/run
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[job] $(date -u +%T) $*"; }
save() { cp -r "$OUT"/. "$SAVE"/ 2>/dev/null; rm -rf "$SAVE"/round1/adapter; log "saved: $(ls "$SAVE" | tr '\n' ' ')"; }
fail() { log "FAILED: $1"; tail -30 "$OUT/$1.log" 2>/dev/null | sed 's/^/[job]   /'; save; exit 1; }
step() {  # step NAME CMD... ; logs to $OUT/NAME.log, echoes progress lines
  local name=$1; shift
  log "step $name"
  "$@" > "$OUT/$name.log" 2>&1 &
  local pid=$!
  tail -n +1 -F "$OUT/$name.log" 2>/dev/null | grep --line-buffered -E "^\[(sample|train)\]|Traceback|Error|out of memory" &
  local tailer=$!
  wait "$pid"; local rc=$?
  sleep 2; kill "$tailer" 2>/dev/null; pkill -f "tail -n +1 -F $OUT/$name.log" 2>/dev/null
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ' | sed 's/^/[job] gpu mem after: /'; echo
  [ $rc -eq 0 ] || fail "$name"
  save
}

if [ -z "${SKIP_SETUP:-}" ]; then
log "setup"
pip install -q uv
uv venv /opt/venv --python 3.12 -q && source /opt/venv/bin/activate
uv pip install -q vllm==0.32.0 transformers==5.19.0 peft==0.21.2 accelerate hf_transfer "huggingface_hub[cli]" numpy scipy \
  || { log "FAILED: install"; exit 1; }
python -c "import vllm, transformers, peft, torch; print('[job] vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__, torch.cuda.device_count(), 'gpus')"

log "download (cache: $MODEL)"
hf download meta-models/Muse-Glimmer-30B --local-dir "$MODEL" > "$OUT/download.log" 2>&1 || fail download
du -sh "$MODEL"
fi

cp "$CODE/aws_plan.json" "$OUT/plan.json"
echo '{"base_train": {"n": 4, "per_level": 25}}' > /opt/run/overrides.json
TP=${BT_NUM_GPUS:-2}
step base_test python -m pilot.sample --model "$MODEL" --tasks "$CODE/tasks/test_sub.jsonl" --out "$OUT/base_test" --n 4 --tp "$TP" --chunk 25
step base_train python -m pilot.sample --model "$MODEL" --tasks "$CODE/tasks/train.jsonl" --out "$OUT/base_train" --n 4 --tp "$TP" \
  --levels 5,10,20 --chunk 25
step train python -m pilot.train --model "$MODEL" --samples "$OUT/base_train/samples.jsonl" --out "$OUT/round1" --cap 2 --max-examples 400
cp -r "$OUT/round1/adapter" "$SAVE/adapter"
if ! ( step r1_test python -m pilot.sample --model "$MODEL" --tasks "$CODE/tasks/test_sub.jsonl" --out "$OUT/r1_test" --n 4 \
          --tp "$TP" --chunk 25 --lora "$OUT/round1/adapter" ); then
  log "LoRA serving failed; merging adapter into base weights"
  rm -rf "$OUT/r1_test"
  step merge env CUDA_VISIBLE_DEVICES= python -m pilot.merge "$MODEL" "$OUT/round1/adapter" /opt/merged
  step r1_test python -m pilot.sample --model /opt/merged --tasks "$CODE/tasks/test_sub.jsonl" --out "$OUT/r1_test" --n 4 \
    --tp "$TP" --chunk 25
fi
python -m pilot.report "$OUT" > "$OUT/report.log" 2>&1 || fail report
save
sed 's/^/[report] /' "$OUT/report.md"
log "done"
