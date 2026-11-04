#!/bin/bash
# Run 13 tests once training ends: the large-body tests, regressions, then the 747 sweep. Watchdog on the trainer log.
R=/Users/vishnuv/Documents/Projects/samgo; CB=$HOME/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0
PY=$HOME/.venvs/fwcookbook/bin/python
RUN=${RUN:-objects-r13}; LOG=$R/runs/train/$RUN.log
cd $R
while pgrep -f "run-id $RUN " >/dev/null; do
  age=$(( $(date +%s) - $(stat -f %m $LOG) ))
  if [ $age -gt 900 ]; then echo "watchdog: no progress for ${age}s $(date +%H:%M)"; pkill -TERM -f "run-id $RUN "; sleep 120; pkill -KILL -f "run-id $RUN "; $PY runs/fw_delete_sessions.py $LOG; break; fi
  sleep 60
done
grep -E '"event": "(promoted|sampler_saved|done)"' $LOG | cut -c1-300 | sed 's/^/- /' >> fireworks-training-runs/objects-r13/run.md
P=${P_OVERRIDE:-}; [ -n "$P" ] || P=$(grep '"event": "sampler_saved"' $LOG | tail -1 | python3 -c "import sys, json; t=sys.stdin.read(); print(json.loads(t)['path'] if t.strip() else '')")
[ -n "$P" ] || { echo "no sampler saved; see $LOG"; tail -3 $LOG; exit 1; }
echo "trained: $P $(date +%H:%M)"; echo "- Tested sampler: $P" >> $R/fireworks-training-runs/objects-r13/run.md
export PYTHONPATH="$R:$CB"
for t in flaps rotors vrotors blades; do
  (cd $R && $PY -u -m samgo.eval_primitives runs/audit/r13-$t-large --models "r13=$P" > runs/audit/r13-$t-large/r13.log 2>&1; echo "$t-large done $(date +%H:%M)") &
done
wait
for t in r6-primitives r9-sheets r10-rotors r11-vrotors r12-blades; do
  (cd $R && $PY -u -m samgo.eval_primitives runs/audit/$t --models "r13=$P" > runs/audit/$t/r13.log 2>&1; echo "$t done $(date +%H:%M)") &
done
(cd $CB && $PY -u -m samgo.eval_objects --out $R/runs/audit/r5-frozen-doors --models "r13=$P" > $R/runs/audit/r5-frozen-doors/r13.log 2>&1; echo "doors done $(date +%H:%M)"; $PY -u -m samgo.eval_objects --out $R/runs/audit/r4-frozen-v2 --models "r13=$P" > $R/runs/audit/r4-frozen-v2/r13.log 2>&1; echo "objects done $(date +%H:%M)") &
(cd $CB && $PY -u -m samgo.eval_running_gear --n 50 --rounds 3 --data data/running_gear_frame --max-tokens 6000 --model "$P" --out $R/runs/eval/r13_cars.json > $R/runs/eval/r13_cars.log 2>&1; echo "cars done $(date +%H:%M)") &
(cd $R && $PY -u -m samgo.granularity runs/s6/boeing747 --model "$P" --spark --out runs/granularity/r13.json > runs/granularity/r13.log 2>&1; echo "sweep done $(date +%H:%M)") &
wait
echo "all tests done $(date +%H:%M)"
