#!/bin/bash
# Wait for the first baseline chunk of a slow run, then hand the box to phase2.sh via SSM.
ID=${1:?instance id}
RUN=${RUN:-pilot-002}
LIVE=s3://samgo-runs-811269446285/live/$RUN
until aws s3 cp "$LIVE/base_test/progress.json" - 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin)['tasks_done'] >= 100 else 1)" 2>/dev/null; do
  sleep 20
done
echo "chunk 1 landed $(date -u +%T); taking over"
read -r -d '' SCRIPT <<'EOF'
set -x
SAMPLER=$(pgrep -f "python -m pilot.sample" | head -1)
MAIN=$(ps -o ppid= -p "$SAMPLER" | tr -d ' ')
echo "sampler $SAMPLER main $MAIN: $(ps -o args= -p $MAIN)"
wc -l /opt/run/out/base_test/samples.jsonl
kill -9 "$MAIN"
pkill -TERM -f "python -m pilot.sample"; sleep 15; pkill -9 -f "python -m pilot.sample"; pkill -9 -f "vllm" ; sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
aws s3 cp s3://samgo-runs-811269446285/code/pilot-002-overrides.json /opt/run/overrides.json
aws s3 cp s3://samgo-runs-811269446285/code/pilot-002-phase2.sh /opt/run/phase2.sh
nohup bash /opt/run/phase2.sh > /opt/run/out/phase2.log 2>&1 &
sleep 3; pgrep -fl phase2.sh; pgrep -fl "sleep 60" | head -2; shutdown --show 2>&1 | tail -1
EOF
CMD=$(aws ssm send-command --region us-west-2 --instance-ids "$ID" --document-name AWS-RunShellScript \
  --parameters "$(python3 -c 'import json,sys; print(json.dumps({"commands": [sys.argv[1]]}))' "$SCRIPT")" \
  --query Command.CommandId --output text)
sleep 40
aws ssm get-command-invocation --region us-west-2 --command-id "$CMD" --instance-id "$ID" \
  --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
