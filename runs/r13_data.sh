#!/bin/bash
# Run 13 data: the same four generic skills (hinged flap, side rotor, upright rotor, blade cross) on bodies 1.5-2.5x
# larger (samgo.flap_data.sheet_body, SAMGO_BODY_SCALE), plus a frozen large-body test per skill. No object words.
set -u
R=/Users/vishnuv/Documents/Projects/samgo; cd $R
export PYTHONPATH=$R SAMGO_BODY_SCALE=1.5,2.5
PY=$HOME/.venvs/samgo/bin/python
for m in flap_data:flaps rotor_data:rotors vrotor_data:vrotors blade_data:blades; do
  mod=${m%%:*}; name=${m##*:}
  $PY -m samgo.$mod --out data/${name}_large_r13 --test runs/audit/r13-${name}-large --n 1500 --test-n 30 --workers 8 \
    > runs/r13_data_${name}.log 2>&1
  echo "$name done $(date +%H:%M): $(grep -o '"train_rows": [0-9]*' runs/r13_data_${name}.log)"
done
