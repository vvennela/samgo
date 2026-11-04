#!/bin/bash
# Showcase at higher resolution: pipeline builds (no hand-finishing) with run 11's Glimmer, one after another.
R=/Users/vishnuv/Documents/Projects/samgo; CB=$HOME/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0
PY=$HOME/.venvs/fwcookbook/bin/python
P11=vvennela/run-736d451107854752b6ba9b0cb0ad5b78/objects-r11-tuned-d5f4be8e
cd $R; export PYTHONPATH="$R:$CB"; O=runs/demo_r11_xl; mkdir -p $O
run() { n=$1; shift; $PY -u -m samgo.pipeline "$@" --assembly learned --model $P11 -o $O/$n > $O/$n.log 2>&1; echo "$n done $(date +%H:%M) $(tail -12 $O/$n.log | grep -o '"status": "[a-z_]*"' | tail -1)"; }
run helicopter photos/helicopter.jpg --prompt helicopter --static --size 40
run cessna photos/cessna.jpg --prompt airplane --static --size 40
run desk_fan photos/desk_fan.jpg --prompt fan --static --size 40
run fan2 photos/fan2.jpg --prompt fan --static --size 40
run wind_turbine photos/wind_turbine.jpg --prompt "wind turbine" --static --size 40
run camry photos/camry.jpg --prompt car --car --doors --size large --color red
echo "all done $(date +%H:%M)"
