#!/bin/bash
# User-data for an instance launched from a pilot image (model, venv, baseline samples on disk).
# Pulls the latest phase2.sh and train.py, clears the failed attempt, and resumes at training.
RUN=pilot-002
BUCKET=s3://samgo-runs-811269446285
shutdown -h +150   # dead-man until phase2.sh sets its own
echo '{"phase": "resuming_train", "t": "'"$(date -u +%FT%TZ)"'"}' | aws s3 cp - "$BUCKET/live/$RUN/phase.json"
aws s3 rm "$BUCKET/live/$RUN/failure.txt" --quiet
rm -rf /opt/run/out/failure.txt /opt/run/out/round1 /opt/run/out/train.log /opt/run/out/r1_test
aws s3 cp "$BUCKET/code/$RUN-phase2.sh" /opt/run/phase2.sh
aws s3 cp "$BUCKET/code/$RUN-train.py" /opt/run/code/pilot/train.py
START_AT=train MAX_MIN=130 TRAIN_CAP=2 RUN=$RUN bash /opt/run/phase2.sh > /opt/run/out/phase2.log 2>&1
