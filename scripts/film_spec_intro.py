"""The intro on its own (scripts/film.py): the on-camera take, the two Muse Image pictures over "models can draw"
(covering the phone settling), word-by-word captions, the SAMGO title on after the question."""
import json
import os
from pathlib import Path

from PIL import Image

FILM = os.environ.get("FILM_ASSETS", "/private/tmp/claude-501/-Users-vishnuv-Documents-Projects-samgo/"
                      "8e62d17f-cb50-41fb-8609-cb7914a1e9e1/scratchpad/film")
TITLE = os.environ.get("FILM_TITLE", "/private/tmp/claude-501/-Users-vishnuv-Documents-Projects-samgo/"
                       "8e62d17f-cb50-41fb-8609-cb7914a1e9e1/scratchpad/intro/frames_wide")
MI = "/Users/vishnuv/Documents/Projects/samgo/runs/video/muse_image"
C = f"{FILM}/cards"
W, H, FPS = 1920, 1080, 30
BG = (238, 241, 245)
T0 = 0.0
CAMERA = [dict(t=0, x=0, y=0, zoom=1.0)]
FIX = {"drop,": "draw,", "drop": "draw"}
WORDS = [dict(w, w=FIX.get(w["w"], w["w"])) for w in json.loads(Path(f"{FILM}/why_words.json").read_text())]
_at = {w["w"].strip(",.?").lower(): w for w in WORDS}
T_BUT = _at["but"]["t0"]
T_END = WORDS[-1]["t1"]
DURATION = len(list(Path(f"{FILM}/intro2").glob("*.png"))) / FPS
lbl = f"{C}/lbl_drawn.png"
ELEMENTS = [
    dict(src=f"{FILM}/intro2", screen=True, x=960, y=540, w=1920, start=0),
    dict(src=f"{FILM}/bg.png", screen=True, x=960, y=540, w=1920, show=(0, T_BUT), fade=0.25),
    dict(src=f"{MI}/lego_sedan_a.png", screen=True, x=500, y=500, w=920, show=(0, T_BUT), fade=0.25, radius=0.02,
         shadow=True),
    dict(src=f"{MI}/lego_sedan_b.png", screen=True, x=1420, y=500, w=920, show=(0, T_BUT), fade=0.25, radius=0.02,
         shadow=True),
    dict(src=lbl, screen=True, x=960, y=880, w=Image.open(lbl).width * 0.9, show=(0, T_BUT), fade=0.25),
    dict(src=TITLE, screen=True, x=960, y=540, w=1920, start=T_END + 0.15, show=(T_END + 0.15, 999), fade=0.0),
    dict(words=WORDS, y=1000, size=44),
]
