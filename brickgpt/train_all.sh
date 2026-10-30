#!/bin/bash
# Train v1 (BrickGPT replication) then v2 (35-part palette) on Fireworks serverless, keeping the Mac awake.
# The account runs one Glimmer training at a time, so they go in sequence.
#
# The trainer runs from this Mac, so it stops when the Mac goes offline or to sleep (the session then expires
# after ~20 min). Each run is therefore watched: if its metrics log is silent for STALL seconds, or it exits
# early, it is stopped, and once Fireworks is reachable again it is restarted; it resumes from its last saved
# model (a checkpoint every 4,000 rows, or the snapshot its cleanup saves on a clean stop). At most TRIES tries.
REPO=$(cd "$(dirname "$0")/.." && pwd)
COOKBOOK=$HOME/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0
PY=$HOME/.venvs/fwcookbook/bin/python
STALL=1500
TRIES=8
mkdir -p "$REPO/runs/train"
cd "$COOKBOOK" || exit 1
export PYTHONPATH="$REPO:$COOKBOOK"

say() { echo "{\"t\": \"$(date +%H:%M:%S)\", \"event\": \"watchdog\", $1}"; }

online() { [ "$(curl -s -o /dev/null -m 10 -w '%{http_code}' https://api.fireworks.ai)" != "000" ]; }

run() {  # run_id palette train eval
  local id=$1 metrics="$REPO/runs/train/$1/metrics.jsonl"
  for try in $(seq 1 $TRIES); do
    until online; do sleep 60; done
    $PY -u "$REPO/brickgpt/train_serverless.py" --run-id "$id" --palette "$2" --batch 64 --save-every-rows 4000 \
        --train "$3" --eval "$4" &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      local age=$(( $(date +%s) - $(stat -f %m "$metrics" 2>/dev/null || date +%s) ))
      if [ $age -gt $STALL ]; then
        say "\"run\": \"$id\", \"why\": \"stalled\", \"silent_s\": $age, \"try\": $try"
        kill -TERM $pid; sleep 180; kill -KILL $pid 2>/dev/null
      fi
    done
    wait $pid && return 0
    say "\"run\": \"$id\", \"why\": \"exited\", \"try\": $try"
    sleep 60
  done
  say "\"run\": \"$id\", \"why\": \"gave_up\""
  return 1
}

caffeinate -dimsu -w $$ &
run brickgpt-v1 brickgpt data/sft/train_c2.jsonl data/sft/eval_500.jsonl &&
run samgo-v2 v2 data/v2/train_c2.jsonl data/v2/test_c1.jsonl
