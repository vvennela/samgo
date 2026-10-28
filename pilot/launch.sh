#!/bin/bash
# Launch exactly one pilot instance, retrying every WAIT seconds while 4-GPU capacity is short.
# Prints the instance id and exits on the first success; gives up after ROUNDS rounds.
cd "$(dirname "$0")/.."
ROUNDS=${ROUNDS:-60}
WAIT=${WAIT:-120}
RUN=${RUN:?set RUN, e.g. RUN=pilot-002}
USERDATA=$(mktemp)
IMAGE=${IMAGE:-ami-0ca70308d230e8a6e}   # base DLAMI; a pilot image to resume
SRC=${SRC:-pilot/run.sh}               # user-data script
LOG=runs/pilot/launch_$RUN.log
mkdir -p runs/pilot
exec > >(tee -a "$LOG") 2>&1
sed "s/^RUN=.*/RUN=$RUN/" "$SRC" > "$USERDATA"
for round in $(seq 1 "$ROUNDS"); do
  for type in ${TYPES:-g6e.12xlarge g5.12xlarge g6.12xlarge}; do
    out=$(aws ec2 run-instances --region us-west-2 --image-id "$IMAGE" --instance-type "$type" --count 1 \
      --iam-instance-profile Name=SamgoPilotEC2 --instance-initiated-shutdown-behavior stop \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=400,VolumeType=gp3,Iops=6000,Throughput=1000,DeleteOnTermination=true}' \
      --metadata-options HttpTokens=required,HttpEndpoint=enabled \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=samgo-'"$RUN"'},{Key=project,Value=samgo}]' \
                           'ResourceType=volume,Tags=[{Key=project,Value=samgo}]' \
      --user-data "file://$USERDATA" --query 'Instances[0].InstanceId' --output text 2>&1)
    case "$out" in
      i-*) echo "LAUNCHED $out $type round $round $(date -u +%FT%TZ)"; exit 0 ;;
      *) echo "round $round $type: $(echo "$out" | tail -1 | cut -c1-90)" ;;
    esac
  done
  [ "$round" -lt "$ROUNDS" ] && sleep "$WAIT"
done
echo "GAVE UP after $ROUNDS rounds $(date -u +%FT%TZ)"
exit 1
