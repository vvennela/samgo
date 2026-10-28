#!/bin/bash
# Stream a run's live S3 files as events: phase changes, new [grpo] lines, new error lines, instance state.
RUN=${1:?run}; LIVE=s3://samgo-runs-811269446285/live/$RUN; TAG=${2:-samgo-$RUN}
seen=$(mktemp); errs=$(mktemp); last_phase=""; last_state=""
while true; do
  phase=$(aws s3 cp "$LIVE/phase.json" - 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['phase'])" 2>/dev/null)
  [ -n "$phase" ] && [ "$phase" != "$last_phase" ] && { echo "phase: $phase"; last_phase=$phase; }
  aws s3 cp "$LIVE/grpo_lines.txt" - 2>/dev/null | while read -r l; do grep -qxF "$l" "$seen" || { echo "$l" | cut -c1-400; echo "$l" >> "$seen"; }; done
  aws s3 cp "$LIVE/errors.txt" - 2>/dev/null | while read -r l; do grep -qxF "$l" "$errs" || { echo "ERROR: $l" | cut -c1-300; echo "$l" >> "$errs"; }; done
  state=$(aws ec2 describe-instances --region us-west-2 --filters Name=tag:Name,Values=$TAG \
          --query 'sort_by(Reservations[].Instances[],&LaunchTime)[-1].State.Name' --output text 2>/dev/null)
  [ -n "$state" ] && [ "$state" != "None" ] && [ "$state" != "$last_state" ] && { echo "instance: $state"; last_state=$state; }
  case "$phase" in done|failed*|no_code_shutdown) [ "$state" = stopped ] && { echo "RUN ENDED: $phase"; exit 0; } ;; esac
  sleep 60
done
