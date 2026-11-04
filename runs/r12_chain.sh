#!/bin/bash
# Run 12: the blade cross (a real-LEGO rotor: axle ending in a round hub plate, two crossed blade plates, a cap) on
# generic bodies, on top of run 11's data. Mix -> train -> tests.
set -u
R=/Users/vishnuv/Documents/Projects/samgo; CB=$HOME/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0
PY=$HOME/.venvs/fwcookbook/bin/python
cd $R
export PYTHONPATH="$R:$CB"
mkdir -p data/mix_r12
python3 - <<'PY'
import json, random
r11 = [json.loads(l) for l in open("data/mix_r11/train.jsonl")]
bl = [{"messages": json.loads(l)["messages"]} for l in open("data/blades_r12/train.jsonl")]
mix = r11 + bl
random.Random(11).shuffle(mix)
open("data/mix_r12/train.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in mix))
open("data/mix_r12/val.jsonl", "w").write(open("data/mix_r11/val.jsonl").read())
print("mix", len(mix), "run11", len(r11), "blades", len(bl), flush=True)
PY
LOG=$R/runs/train/objects-r12.log
(cd $CB && caffeinate -i $PY -u $R/brickgpt/train_serverless.py --run-id objects-r12 --palette none --batch 64 --epochs 2 --max-seq-len 24576 --train $R/data/mix_r12/train.jsonl --eval $R/data/mix_r12/val.jsonl --n-eval 0 --warmup-rows 256 --save-every-rows 2000 > $LOG 2>&1) &
sleep 60
$R/runs/r12_tests.sh
