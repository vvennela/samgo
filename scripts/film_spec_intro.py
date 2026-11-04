"""The intro (scripts/film.py): the on-camera take as shot (sky, then the face), the two Muse Image pictures over
"we all know that models can draw", word-by-word captions, the SAMGO title on "Introducing SAMgo"."""
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
FIX = {"Samgo.": "SAMgo.", "Samgo": "SAMgo", "Lego.": "LEGO."}
WORDS = [dict(w, w=FIX.get(w["w"], w["w"])) for w in json.loads(Path(f"{FILM}/why_words.json").read_text())]
idx = {w["w"].strip(",.?").lower() + str(i): w for i, w in enumerate(WORDS)}
first = lambda word: next(w for w in WORDS if w["w"].strip(",.?").lower() == word)
T_WE = first("we")["t0"]
T_BUT = first("but")["t0"]
T_INTRO = first("introducing")["t0"]
T_TITLE = 17.08                     # the onset of "SAMgo" after "Introducing" (from the audio)
T_DESC = 17.94                      # "SAMgo allows you to..."
DURATION = len(list(Path(f"{FILM}/intro2").glob("*.png"))) / FPS
lbl = f"{C}/lbl_drawn.png"
pics = (T_WE - 0.1, T_BUT)
ELEMENTS = [
    dict(src=f"{FILM}/intro2", screen=True, x=960, y=540, w=1920, start=0),
    dict(src=f"{FILM}/bg.png", screen=True, x=960, y=540, w=1920, show=pics, fade=0.3),
    dict(src=f"{MI}/lego_sedan_a.png", screen=True, x=500, y=500, w=920, show=pics, fade=0.3, radius=0.02, shadow=True),
    dict(src=f"{MI}/lego_sedan_b.png", screen=True, x=1420, y=500, w=920, show=(pics[0] + 0.15, pics[1]), fade=0.3,
         radius=0.02, shadow=True),
    dict(src=lbl, screen=True, x=960, y=880, w=Image.open(lbl).width * 0.9, show=pics, fade=0.3),
    dict(src=TITLE, screen=True, x=960, y=540, w=1920, start=T_TITLE - 0.05, show=(T_TITLE - 0.05, T_DESC + 1.2),
         fade=0.35),
    dict(words=[w for w in WORDS if not (T_INTRO - 0.05 <= w["t0"] < T_DESC - 0.05)], y=1000, size=44),
]
