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
seg() { FILM_NO_TRANSITIONS=1 $PY scripts/film.py scripts/film_spec.py $W/$1.mp4 --from $2 --to $3 --workers 10 --blur 3 >/dev/null; echo "segment $1 $2-$3"; }
# one clip per shot; every cut between them is a stud dissolve (scripts/film_join.py)
seg 01_intro 0 9.6            # "we know models can draw (Muse Image's picture) ... but can they build" + the title
seg 03_photo 10.1 12.6         # one photo
seg 04_full 44.1 51.0         # full scale: 144,753 pieces, every piece connected
seg 05_down 51.5 56.0         # -> 293 pieces
seg 06_over 56.5 62.4         # the 293 over SAM 3D's mesh: same shape, real LEGO
seg 07_work 62.9 85.3         # Spark and Glimmer make it a car: wheels and axles, glass, doors
seg 09_manual 92.3 99.7       # the certified manual, flipped
seg 11_today 128.8 132.8      # today LEGO, next real CAD
seg 12_nums 133.3 138.8       # the numbers
seg 13_end 140.9 144.5        # the end card
ffmpeg -v error -y -framerate 30 -i $F/drawing/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 $W/02_drawing.mp4
ffmpeg -v error -y -i "$STUDIO" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 18 -an $W/08_studio.mp4
ffmpeg -v error -y -i "$TOUR" -filter:v "setpts=PTS/3,fps=30" -c:v libx264 -pix_fmt yuv420p -crf 18 -an $W/10_tour.mp4
$PY scripts/film_join.py $W/video.mp4 $W/01_intro.mp4 $W/02_drawing.mp4 $W/03_photo.mp4 $W/04_full.mp4 $W/05_down.mp4 \
  $W/06_over.mp4 $W/07_work.mp4 $W/08_studio.mp4 $W/09_manual.mp4 $W/10_tour.mp4 $W/11_today.mp4 $W/12_nums.mp4 $W/13_end.mp4
TOTAL=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $W/video.mp4)
# audio: the intro's own sound, the music bed under everything
$PY scripts/film_music.py $W/music.wav $TOTAL >/dev/null
INTRO_AUDIO=${INTRO_AUDIO:-$HOME/Downloads/samgo_intro_mock_wide.mov}
ffmpeg -v error -y -i $W/video.mp4 -i $W/music.wav -i "$INTRO_AUDIO" -filter_complex \
  "[1:a]volume=0.9[m];[2:a]atrim=0:9.6,apad[v];[m][v]amix=inputs=2:duration=first:normalize=0[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k "$OUT"
echo "demo -> $OUT ($TOTAL s)"
