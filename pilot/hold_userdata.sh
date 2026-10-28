#!/bin/bash
# User-data for a capacity grab from the pilot image: hold the box for code that may not be uploaded yet.
# Waits up to WAIT_MIN for $BUCKET/code/$RUN-run.sh, then runs it (it sets its own dead-man); else shuts down.
RUN=grpo-aws
BUCKET=s3://samgo-runs-811269446285
WAIT_MIN=60
shutdown -h +$((WAIT_MIN + 15))
mkdir -p /opt/run/out
echo '{"phase": "holding", "t": "'"$(date -u +%FT%TZ)"'"}' | aws s3 cp - "$BUCKET/live/$RUN/phase.json"
for i in $(seq 1 $((WAIT_MIN * 2))); do
  if aws s3 cp "$BUCKET/code/$RUN-run.sh" /opt/run/run.sh 2>/dev/null; then
    RUN=$RUN bash /opt/run/run.sh > /opt/run/out/run_sh.log 2>&1
    exit 0
  fi
  sleep 30
done
echo '{"phase": "no_code_shutdown", "t": "'"$(date -u +%FT%TZ)"'"}' | aws s3 cp - "$BUCKET/live/$RUN/phase.json"
shutdown -h now
