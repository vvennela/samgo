"""The launch film's layout and camera (scripts/film.py). World coordinates are pixels at zoom 1; the camera's
(x, y) is the world point at the centre of the frame. Times are seconds; re-time to the narration by moving the
keyframes and the clips' start times.

Assets live in FILM (env FILM_ASSETS): intro/ (the on-camera intro frames), camry/ (the Camry build then motion,
transparent frames from scripts/film_tiles.py), photo.png.
"""
import os

FILM = os.environ.get("FILM_ASSETS", "/private/tmp/claude-501/-Users-vishnuv-Documents-Projects-samgo/"
                      "8e62d17f-cb50-41fb-8609-cb7914a1e9e1/scratchpad/film")
STEPS = "/Users/vishnuv/Documents/Projects/samgo/runs/video/steps"

W, H, FPS = 1920, 1080, 30
BG = (238, 241, 245)          # a soft, cool off-white
DOT = (216, 222, 230)         # the stud-dot grid
GRID = 64
T0 = 0.0

CAMRY_AT = 17.0               # the Camry starts building as the camera arrives

ELEMENTS = [
    # 1. the on-camera intro, a tile on the canvas once the camera pulls back
    dict(src=f"{FILM}/intro", x=0, y=0, w=1920, start=0, radius=0.012, shadow=True),
    # 2. the photo, printed on the canvas
    dict(src=f"{STEPS}/1_photo.png", x=2750, y=0, w=1500, radius=0.018, shadow=True),
    # 3. the LEGO Camry: builds itself, then the doors swing and the wheels turn
    dict(src=f"{FILM}/camry", x=4550, y=0, w=1920, start=CAMRY_AT),
]

CAMERA = [
    dict(t=0.0, x=0, y=0, zoom=1.0),
    dict(t=9.4, x=0, y=0, zoom=1.0),
    dict(t=11.8, x=1250, y=0, zoom=0.42),                 # pull back: the intro is a tile, the photo waits beside it
    dict(t=13.8, x=2750, y=0, zoom=1.08),                 # glide in to the photo
    dict(t=16.0, x=2780, y=10, zoom=1.16, ease="linear"), # slow push while it is named
    dict(t=18.4, x=4550, y=40, zoom=1.35, arc=0.25),      # over to the Camry, pieces already falling
    dict(t=31.0, x=4550, y=30, zoom=1.5, ease="linear"), # the build completes
    dict(t=35.0, x=4550, y=30, zoom=1.58, ease="linear"), # doors swing, wheels turn
    dict(t=38.5, x=3650, y=0, zoom=0.62),                  # pull back: the photo and its LEGO model side by side
    dict(t=40.0, x=3650, y=0, zoom=0.6, ease="linear"),
]
DURATION = 40.0
