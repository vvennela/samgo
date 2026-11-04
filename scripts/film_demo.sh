#!/bin/bash
# The demo cut: illustration segments rendered from scripts/film_spec.py, joined by fades with the real product
# footage (SAMgo Studio replaying the logged live run, and the screen-recorded tour of the 3D viewer), music under.
#   bash scripts/film_demo.sh <studio_show.mp4> <tour.mp4> <out.mp4>
set -e
cd "$(dirname "$0")/.."
STUDIO=$1; TOUR=$2; OUT=$3
F=${FILM_ASSETS:-/private/tmp/claude-501/-Users-vishnuv-Documents-Projects-samgo/8e62d17f-cb50-41fb-8609-cb7914a1e9e1/scratchpad/film}
W=$F/demo; mkdir -p $W
PY=$HOME/.venvs/samgo/bin/python
seg() { $PY scripts/film.py scripts/film_spec.py $W/$1.mp4 --from $2 --to $3 --workers 10 --blur 3 >/dev/null; echo "segment $1 $2-$3"; }
seg 1_intro 0 9.6            # "we know models can draw (Muse Image's picture) ... but can they build" + the title
seg 3_scale 43.6 62.4        # full scale 144,753 -> 293 pieces -> the 293 over SAM 3D's mesh
seg 4_skill 85.3 91.8        # what Glimmer learned: hinges, before -> now
seg 5_manual 91.8 99.7       # the certified manual, flipped
seg 7_close 127.5 144.5      # today LEGO, next real CAD; the numbers; the end card
# the real footage, normalised to 30 fps; the tour at 2x
ffmpeg -v error -y -i "$STUDIO" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 18 -an $W/2_studio.mp4
ffmpeg -v error -y -i "$TOUR" -filter:v "setpts=PTS/2,fps=30" -c:v libx264 -pix_fmt yuv420p -crf 18 -an $W/6_tour.mp4
# join with 0.4 s fades
FILES=($W/1_intro.mp4 $W/2_studio.mp4 $W/3_scale.mp4 $W/4_skill.mp4 $W/5_manual.mp4 $W/6_tour.mp4 $W/7_close.mp4)
IN=""; FG=""; OFF=0; prev="0:v"
for i in "${!FILES[@]}"; do IN="$IN -i ${FILES[$i]}"; done
for i in $(seq 1 $((${#FILES[@]} - 1))); do
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 ${FILES[$((i-1))]})
  OFF=$(python3 -c "print(round($OFF + $d - 0.4, 3))")
  FG="$FG[$prev][$i:v]xfade=transition=fade:duration=0.4:offset=$OFF[v$i];"; prev="v$i"
done
ffmpeg -v error -y $IN -filter_complex "${FG%;}" -map "[$prev]" -c:v libx264 -pix_fmt yuv420p -crf 18 $W/video.mp4
TOTAL=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $W/video.mp4)
# audio: the intro's own sound, the music bed under everything
$PY scripts/film_music.py $W/music.wav $TOTAL >/dev/null
INTRO_AUDIO=${INTRO_AUDIO:-$HOME/Downloads/samgo_intro_mock_wide.mov}
ffmpeg -v error -y -i $W/video.mp4 -i $W/music.wav -i "$INTRO_AUDIO" -filter_complex \
  "[1:a]volume=0.9[m];[2:a]atrim=0:9.6,apad[v];[m][v]amix=inputs=2:duration=first:normalize=0[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k "$OUT"
echo "demo -> $OUT ($TOTAL s)"
