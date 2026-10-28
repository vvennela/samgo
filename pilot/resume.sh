#!/bin/bash
# Restart a stopped pilot instance (retrying for capacity) and run phase2.sh on it via SSM.
ID=${1:?instance id}
RUN=${RUN:-pilot-002}
ROUNDS=${ROUNDS:-30}
LIVE=s3://samgo-runs-811269446285/live/$RUN
for round in $(seq 1 "$ROUNDS"); do
  out=$(aws ec2 start-instances --region us-west-2 --instance-ids "$ID" --query 'StartingInstances[0].CurrentState.Name' --output text 2>&1)
  case "$out" in pending|running) echo "STARTED round $round $(date -u +%T)"; break ;; esac
  echo "round $round: $(echo "$out" | tail -1 | cut -c1-80)"
  [ "$round" -eq "$ROUNDS" ] && { echo "GAVE UP $(date -u +%T)"; exit 1; }
  sleep 120
done
aws ec2 wait instance-running --region us-west-2 --instance-ids "$ID"
until [ "$(aws ssm describe-instance-information --region us-west-2 --filters Key=InstanceIds,Values="$ID" \
          --query 'InstanceInformationList[0].PingStatus' --output text)" = "Online" ]; do sleep 10; done
echo '{"phase": "resuming_train", "t": "'"$(date -u +%FT%TZ)"'"}' | aws s3 cp - "$LIVE/phase.json"
aws s3 rm "$LIVE/failure.txt" --quiet
read -r -d '' SCRIPT <<'EOF'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
rm -rf /opt/run/out/failure.txt /opt/run/out/round1 /opt/run/out/train.log
aws s3 cp s3://samgo-runs-811269446285/code/pilot-002-phase2.sh /opt/run/phase2.sh
aws s3 cp s3://samgo-runs-811269446285/code/pilot-002-train.py /opt/run/code/pilot/train.py
grep -c balanced /opt/run/code/pilot/train.py
START_AT=train MAX_MIN=110 TRAIN_CAP=2 nohup bash /opt/run/phase2.sh > /opt/run/out/phase2.log 2>&1 &
sleep 5; pgrep -fl phase2.sh; cat /run/systemd/shutdown/scheduled 2>/dev/null | head -2
EOF
CMD=$(aws ssm send-command --region us-west-2 --instance-ids "$ID" --document-name AWS-RunShellScript \
  --parameters "$(python3 -c 'import json,sys; print(json.dumps({"commands": [sys.argv[1]]}))' "$SCRIPT")" \
  --query Command.CommandId --output text)
sleep 15
aws ssm get-command-invocation --region us-west-2 --command-id "$CMD" --instance-id "$ID" \
  --query '[Status,StandardOutputContent]' --output text
