#!/bin/bash
# Run 11 tests in parallel once training ends: unseen bodies (r6 primitives, r8 sheets), regressions, then the 747.
R=/Users/vishnuv/Documents/Projects/samgo; CB=$HOME/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0
PY=$HOME/.venvs/fwcookbook/bin/python
LOG=$R/runs/train/objects-r12.log
cd $R
while pgrep -f "train_serverless.py --run-id objects-r12" >/dev/null; do
  age=$(( $(date +%s) - $(stat -f %m $LOG) ))
  if [ $age -gt 900 ]; then echo "watchdog: no progress for ${age}s $(date +%H:%M)"; pkill -TERM -f "train_serverless.py --run-id objects-r12"; sleep 120; pkill -KILL -f "train_serverless.py --run-id objects-r12"; break; fi
  sleep 60
done
P=${P_OVERRIDE:-}; [ -n "$P" ] || P=$(grep '"event": "sampler_saved"' $LOG | tail -1 | python3 -c "import sys, json; t=sys.stdin.read(); print(json.loads(t)['path'] if t.strip() else '')")
[ -n "$P" ] || { echo "no sampler saved; see $LOG"; tail -3 $LOG; exit 1; }
echo "trained: $P $(date +%H:%M)"; echo "- Tested sampler: $P" >> $R/fireworks-training-runs/objects-r12/run.md
export PYTHONPATH="$R:$CB"
(cd $R && $PY -u -m samgo.eval_primitives runs/audit/r6-primitives --models "r12=$P" > runs/audit/r6-primitives/r12.log 2>&1; echo "primitives done $(date +%H:%M)") &
(cd $R && $PY -u -m samgo.eval_primitives runs/audit/r9-sheets --models "r12=$P" > runs/audit/r9-sheets/r12.log 2>&1; echo "sheets done $(date +%H:%M)") &
(cd $R && $PY -u -m samgo.eval_primitives runs/audit/r11-vrotors --models "r12=$P" > runs/audit/r11-vrotors/r12.log 2>&1; echo "vrotors done $(date +%H:%M)") &
(cd $R && $PY -u -m samgo.eval_primitives runs/audit/r12-blades --models "r12=$P" > runs/audit/r12-blades/r12.log 2>&1; echo "blades done $(date +%H:%M)") &
(cd $R && $PY -u -m samgo.eval_primitives runs/audit/r10-rotors --models "r12=$P" > runs/audit/r10-rotors/r12.log 2>&1; echo "rotors done $(date +%H:%M)") &
(cd $R && $PY -u -m samgo.granularity runs/s6/boeing747 --model "$P" --spark --out runs/granularity/r12.json > runs/granularity/r12.log 2>&1; echo "sweep done $(date +%H:%M)") &
(cd $CB && $PY -u -m samgo.eval_objects --out $R/runs/audit/r5-frozen-doors --models "r12=$P" > $R/runs/audit/r5-frozen-doors/r12.log 2>&1; echo "doors done $(date +%H:%M)"; $PY -u -m samgo.eval_objects --out $R/runs/audit/r4-frozen-v2 --models "r12=$P" > $R/runs/audit/r4-frozen-v2/r12.log 2>&1; echo "objects done $(date +%H:%M)") &  # serial: in parallel they hit service errors
(cd $CB && $PY -u -m samgo.eval_running_gear --n 50 --rounds 3 --data data/running_gear_frame --max-tokens 6000 --model "$P" --out $R/runs/eval/r12_cars.json > $R/runs/eval/r12_cars.log 2>&1; echo "cars done $(date +%H:%M)") &
wait
echo "all tests done $(date +%H:%M)"
