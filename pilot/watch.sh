#!/bin/bash
# Watch a pilot run. Polls every 30 s; exits at once on trouble, otherwise after PERIOD seconds.
# Exit 0 = routine check, 2 = alert (failure, done, instance stopped, stalled, or GPUs idle).
ID=${1:?instance id}
RUN=${RUN:-pilot-002}
PERIOD=${PERIOD:-600}
B=s3://samgo-runs-811269446285/$RUN
start=$(date +%s)
idle_since=""

alert() { echo "ALERT: $*"; "$(dirname "$0")/status.sh" "$ID"; exit 2; }

while [ $(( $(date +%s) - start )) -lt "$PERIOD" ]; do
  state=$(aws ec2 describe-instances --region us-west-2 --instance-ids "$ID" \
            --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
  phase=$(aws s3 cp "$B/phase.json" - 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['phase'])" 2>/dev/null)
  case "$phase" in failed*) alert "phase $phase";; done*) alert "run finished: $phase";; esac
  case "$state" in stopping|stopped|shutting-down|terminated) alert "instance $state (phase $phase)";; esac

  # Heartbeat: gpu.log is appended every minute while the box is alive.
  last=$(aws s3 cp "$B/gpu.log" - 2>/dev/null | tail -1 | cut -d' ' -f1)
  if [ -n "$last" ]; then
    age=$(( $(date -u +%s) - $(date -u -j -f "%Y-%m-%d %H:%M:%S" "$(date -u +%Y-%m-%d) $last" +%s 2>/dev/null || date -u +%s) ))
    [ "$age" -gt 300 ] && [ "$age" -lt 80000 ] && alert "no heartbeat for ${age}s (phase $phase)"
    # GPUs idle for 10+ minutes while they should be working.
    case "$phase" in baseline_*|train|round1_test*)
      busy=$(aws s3 cp "$B/gpu.log" - 2>/dev/null | tail -4 | awk -F', ' '{gsub(/ %/,"",$2); if ($2+0 > 5) b=1} END {print b+0}')
      if [ "$busy" = "0" ]; then
        idle_since=${idle_since:-$(date +%s)}
        [ $(( $(date +%s) - idle_since )) -gt 600 ] && alert "GPUs idle 10+ min in $phase"
      else
        idle_since=""
      fi ;;
    esac
  fi
  sleep 30
done
echo "ROUTINE CHECK"
"$(dirname "$0")/status.sh" "$ID"
exit 0
