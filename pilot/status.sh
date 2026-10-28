#!/bin/bash
# One monitoring check: instance state plus everything the run has synced to S3.
ID=${1:?instance id}
RUN=${RUN:-pilot-002}
B=s3://samgo-runs-811269446285/live/$RUN
echo "== $(date -u +%T) instance $(aws ec2 describe-instances --region us-west-2 --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text)"
get() { aws s3 cp "$B/$1" - 2>/dev/null; }
echo "== phase: $(get phase.json)"
echo "== run.log (tail)"; get run.log | tail -8
for p in base_test base_train r1_test; do
  prog=$(get $p/progress.json); [ -n "$prog" ] && echo "== $p progress: $(echo "$prog" | tr -d '\n ' | cut -c1-400)"
done
log=$(get round1/train_log.jsonl); [ -n "$log" ] && { echo "== train: $(echo "$log" | wc -l | tr -d ' ') steps"; echo "$log" | tail -2; }
echo "== gpu (latest)"; get gpu.log | tail -4
echo "== errors"; get errors.txt | tail -6; get failure.txt | head -20
