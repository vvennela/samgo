#!/bin/bash
# Report a Fireworks SFT job's appearance and every state/progress change (read-only).
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
F=$(command -v firectl || echo /opt/homebrew/Cellar/firectl/64/bin/firectl)
JOB=${1:-brickgpt-glimmer-v1}; last=""
while true; do
  out=$($F --api-key "$FIREWORKS_API_KEY" -a vvennela sftj get "$JOB" 2>&1)
  if echo "$out" | grep -qi "not found\|NotFound"; then line="not started"; else
    line=$(echo "$out" | grep -E -i "^(State|Status|Progress|Epoch|Output Model|Error|Message)" | tr '\n' ' ' | tr -s ' ')
  fi
  [ "$line" != "$last" ] && { echo "$(date +%H:%M) $JOB: $line" | cut -c1-400; last=$line; }
  echo "$line" | grep -q -E "COMPLETED|FAILED|CANCELLED" && exit 0
  sleep 120
done
