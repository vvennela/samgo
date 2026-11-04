#!/bin/bash
# Run 13: mechanisms on large bodies. Mix -> train (warm start from run 11, else from scratch) -> tests.
set -u
R=/Users/vishnuv/Documents/Projects/samgo; CB=$HOME/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0
PY=$HOME/.venvs/fwcookbook/bin/python
cd $R
export PYTHONPATH="$R:$CB"
mkdir -p data/mix_r13_warm data/mix_r13
python3 - <<'PY'
import json, random
large = []
for n in ("flaps", "rotors", "vrotors", "blades"):
    large += [{"messages": json.loads(l)["messages"]} for l in open(f"data/{n}_large_r13/train.jsonl")]
large = [r for r in large if sum(len(m["content"]) for m in r["messages"]) < 68000]  # fits 24,576 tokens
bl = [{"messages": json.loads(l)["messages"]} for l in open("data/blades_r12/train.jsonl")]
r11 = [json.loads(l) for l in open("data/mix_r11/train.jsonl")]
r12 = [json.loads(l) for l in open("data/mix_r12/train.jsonl")]
rng = random.Random(13)
warm = large + bl + rng.sample(r11, 4000)
full = r12 + large
for mix in (warm, full):
    rng.shuffle(mix)
for d, mix in (("data/mix_r13_warm", warm), ("data/mix_r13", full)):
    open(f"{d}/train.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in mix))
    open(f"{d}/val.jsonl", "w").write(open("data/mix_r11/val.jsonl").read())
print("large", len(large), "warm mix", len(warm), "full mix", len(full), flush=True)
PY
train() {  # run-id data
  (cd $CB && caffeinate -i $PY -u $R/brickgpt/train_serverless.py --run-id $1 --palette none --batch 64 --epochs 2 \
     --max-seq-len 24576 --train $R/data/$2/train.jsonl --eval $R/data/$2/val.jsonl --n-eval 0 --warmup-rows 256 \
     --save-every-rows 2000 > $R/runs/train/$1.log 2>&1)
}
# warm start: the trainer loads state.json's adapter (run 11's last) and counts rows from 0
mkdir -p runs/train/objects-r13
[ -f runs/train/objects-r13/state.json ] || echo '{"adapter_path": "accounts/vvennela/models/objects-r11-s30720", "rows": 0, "step": 0, "batch": 64}' > runs/train/objects-r13/state.json
train objects-r13 mix_r13_warm &
RUN=objects-r13
while pgrep -f "run-id objects-r13 " >/dev/null && ! grep -q '"event": "step"' runs/train/objects-r13.log; do sleep 30; done
if ! grep -q '"event": "step"' runs/train/objects-r13.log; then
  echo "warm start failed $(date +%H:%M): $(grep -v Warn runs/train/objects-r13.log | tail -2 | cut -c1-200)"
  $PY runs/fw_delete_sessions.py runs/train/objects-r13.log
  RUN=objects-r13s
  train objects-r13s mix_r13 &
  sleep 60
fi
echo "training $RUN $(date +%H:%M)"
echo "- Training run id: $RUN (started $(date '+%b %d %H:%M'))" >> fireworks-training-runs/objects-r13/run.md
RUN=$RUN $R/runs/r13_tests.sh
