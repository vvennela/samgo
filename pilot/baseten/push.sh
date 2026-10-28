#!/bin/bash
# Stage code, tasks and the AWS plan into a clean dir and push the Baseten job.
cd "$(dirname "$0")/../.."
STAGE=${STAGE:-$(mktemp -d)}
mkdir -p "$STAGE/tasks"
cp -r samgo pilot "$STAGE/" && cp pilot/baseten/job.sh pilot/baseten/config.py "$STAGE/"
cp runs/pilot/tasks/test_sub.jsonl runs/pilot/tasks/train.jsonl "$STAGE/tasks/"
cp runs/pilot/pilot-002/aws/plan.json "$STAGE/aws_plan.json"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
echo "staged $(du -sh "$STAGE" | cut -f1) in $STAGE"
cd "$STAGE" && truss train push config.py --non-interactive
