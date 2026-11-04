"""The launch film's layout and camera (scripts/film.py): one canvas, one camera, no cuts.

World coordinates are pixels at zoom 1; a camera keyframe's (x, y) is the world point at the centre of the frame.
Placeholder pacing until the narration is in: re-time by moving keyframes, start and pop times.

                       fill (SAM 3D -> greedy slab)   time-lapse (Spark + Glimmer) + chips   the build   manual   parts
                              ^
    intro    photo (SAM 3.1)  ->  the Camry (promise)  ->  checklist
                                  helicopter                747
    ... the machines, superimposed ... known parts -> an assembly that holds -> CAD ... everything, numbers, end.
"""
import os
from pathlib import Path

from PIL import Image

FILM = os.environ.get("FILM_ASSETS", "/private/tmp/claude-501/-Users-vishnuv-Documents-Projects-samgo/"
                      "8e62d17f-cb50-41fb-8609-cb7914a1e9e1/scratchpad/film")
C = f"{FILM}/cards"
STEPS = "/Users/vishnuv/Documents/Projects/samgo/runs/video/steps"

W, H, FPS = 1920, 1080, 30
BG = (238, 241, 245)
BG_IMAGE = f"{FILM}/bg.png"          # soft warm-white radial gradient, fixed on screen
T0 = 0.0
DURATION = 180.0


def nat(path, k=1.0):
    return Image.open(path).width * k


def fit(path, cx, cy, content_w, **kw):
    """Place a transparent frame by its content: the drawn model's bounding box centred at (cx, cy), content_w wide."""
    im = Image.open(path)
    x0, y0, x1, y1 = im.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    k = content_w / (x1 - x0)
    w = im.width * k
    return dict(src=path, x=cx - ((x0 + x1) / 2 - im.width / 2) * k, y=cy - ((y0 + y1) / 2 - im.height / 2) * k, w=w,
                **kw)


def label(name, x, y, show, k=0.9):
    p = f"{C}/{name}.png"
    return dict(src=p, x=x, y=y, w=nat(p, k), show=show, fade=0.45)


def chip(name, x, y, at, until, k=0.82):
    p = f"{C}/{name}.png"
    return dict(src=p, x=x, y=y, w=nat(p, k), pop=at, show=(at, until), fade=0.35, shadow=True, radius=0.05)


MANUAL = sorted(str(p) for p in Path(f"{FILM}/manual").glob("p-*.png"))
FLIP = f"{FILM}/flip"
SCHEDULE, t, d = [0.0], 0.7, 0.42
for _ in range(56):
    SCHEDULE.append(t); t += d; d = max(0.085, d * 0.91)
SCRIM = f"{FILM}/scrim.png"
if not Path(SCRIM).exists():
    Image.new("RGBA", (1920, 1080), BG + (225,)).save(SCRIM)

# ---------------------------------------------------------------- where things are
PHOTO = (2350, 0)
PROMISE = (4200, 0)
CHECK = (6000, 0)
HELI, B747 = (4200, 1320), (6000, 1320)
FILL = (2350, -1350)
COLLAB = (4250, -1350)
CHIPS_X = 5220
BUILD = (6450, -1350)
BOOK = (8350, -1390)
PARTS = (10000, -1390)
GROUP = (12700, 0)
MFG = (15700, 0)

# ---------------------------------------------------------------- when
COLLAB_AT = 66.0          # the time-lapse starts (scripts/film_collab.py: rings 1.6 s, drops 3.4-10.3 s)
PAINT_AT = COLLAB_AT + 12.5

ELEMENTS = [
    # intro and the promise
    dict(src=f"{FILM}/intro", x=0, y=0, w=1920, start=0, radius=0.012, shadow=True),
    dict(src=f"{STEPS}/1_photo.png", x=PHOTO[0], y=PHOTO[1], w=1500, radius=0.018, shadow=True),
    dict(src=f"{FILM}/sam31_car.png", x=PHOTO[0], y=PHOTO[1], w=1500, radius=0.018, show=(42.3, 50.5), fade=0.5),
    dict(src=f"{FILM}/sam31_glass.png", x=PHOTO[0], y=PHOTO[1], w=1500, radius=0.018, show=(45.3, 50.5), fade=0.5),
    label("lbl_sam31", PHOTO[0], -400, (42.3, 50.5)),
    dict(src=f"{FILM}/camry_move", x=PROMISE[0], y=PROMISE[1], w=1920, start=15.5),
    dict(src=f"{C}/checklist.png", x=CHECK[0], y=CHECK[1], w=1150, shadow=True, radius=0.03),
    dict(src=f"{FILM}/helicopter", x=HELI[0], y=HELI[1], w=1800, start=26.5, loop=True),
    dict(src=f"{FILM}/b747", x=B747[0], y=B747[1], w=1800, start=30.5, loop=True),

    # how it works: SAM 3D and the greedy fill, then Spark and Glimmer edit it into a car
    dict(src=f"{FILM}/fill", x=FILL[0], y=FILL[1], w=1920, start=50.0),
    label("lbl_sam3d", FILL[0], FILL[1] - 520, (50.0, 64.5)),
    dict(src=f"{FILM}/collab", x=COLLAB[0], y=COLLAB[1], w=1920, start=COLLAB_AT, show=(0, PAINT_AT + 0.6), fade=0.6),
    dict(src=f"{FILM}/painted", x=COLLAB[0], y=COLLAB[1], w=1920, start=PAINT_AT, show=(PAINT_AT, 999), fade=0.6),
    chip("chip_spark1", CHIPS_X, -1700, COLLAB_AT + 1.4, COLLAB_AT + 7.2),
    chip("chip_glimmer1", CHIPS_X, -1480, COLLAB_AT + 3.0, COLLAB_AT + 7.2),
    chip("chip_check1", CHIPS_X, -1255, COLLAB_AT + 6.6, COLLAB_AT + 7.2),
    chip("chip_glimmer2", CHIPS_X, -1480, COLLAB_AT + 7.4, COLLAB_AT + 12.3),
    chip("chip_check2", CHIPS_X, -1255, COLLAB_AT + 10.9, COLLAB_AT + 12.3),
    chip("chip_glass", CHIPS_X, -1480, PAINT_AT + 0.2, PAINT_AT + 3.8),
    dict(src=f"{FILM}/camry", x=BUILD[0], y=BUILD[1], w=1920, start=83.5),
    dict(src=FLIP, x=BOOK[0], y=BOOK[1], w=1450, start=100.3, schedule=SCHEDULE, radius=0.012, shadow=True),
    dict(src=MANUAL[57], x=PARTS[0] - 420, y=PARTS[1], w=760, radius=0.02, shadow=True),
    dict(src=MANUAL[58], x=PARTS[0] + 420, y=PARTS[1], w=760, radius=0.02, shadow=True),

    # the machines, superimposed: each lands on top of the last (still models)
    fit(f"{FILM}/b747/00000.png", GROUP[0], GROUP[1] - 200, 1500, pop=117.2, pop_dur=0.55),
    fit(f"{FILM}/turbine/00000.png", GROUP[0] - 980, GROUP[1] - 120, 430, pop=118.0),
    fit(f"{FILM}/helicopter/00000.png", GROUP[0] + 930, GROUP[1] + 40, 560, pop=118.7),
    fit(f"{FILM}/cessna/00000.png", GROUP[0] - 120, GROUP[1] + 300, 640, pop=119.4),
    fit(f"{FILM}/gt1/00000.png", GROUP[0] - 700, GROUP[1] + 470, 560, pop=120.1),
    fit(f"{FILM}/camry_move/00299.png", GROUP[0] + 560, GROUP[1] + 480, 600, pop=120.8),
    dict(src=f"{C}/cap_unseen.png", x=GROUP[0], y=GROUP[1] - 820, w=nat(f"{C}/cap_unseen.png", 0.85), pop=122.0),

    # why it matters
    dict(src=f"{C}/cap_mfg1.png", x=MFG[0], y=MFG[1] - 820, w=nat(f"{C}/cap_mfg1.png"), pop=130.6),
    dict(src=MANUAL[57], x=MFG[0] - 930, y=MFG[1] + 250, w=520, radius=0.02, shadow=True, pop=135.4),
    dict(src=FLIP, x=MFG[0] - 60, y=MFG[1] + 250, w=900, start=135.8, schedule=SCHEDULE, radius=0.012, shadow=True,
         pop=135.8),
    fit(f"{FILM}/camry_move/00299.png", MFG[0] + 880, MFG[1] + 270, 620, pop=136.2),
    dict(src=f"{C}/cap_mfg2.png", x=MFG[0], y=MFG[1] + 1250, w=nat(f"{C}/cap_mfg2.png"), pop=147.2),
    dict(src=f"{C}/cap_mfg3.png", x=MFG[0], y=MFG[1] + 2100, w=nat(f"{C}/cap_mfg3.png"), pop=153.2),

    # over everything: the numbers, the end card (fixed on screen)
    dict(src=SCRIM, x=960, y=540, w=1920, screen=True, show=(163.8, 181), fade=1.2),
    dict(src=f"{C}/num0.png", x=360, y=470, w=nat(f"{C}/num0.png", 0.68), screen=True, pop=164.6, show=(164.6, 171.4)),
    dict(src=f"{C}/num0_sub.png", x=360, y=640, w=nat(f"{C}/num0_sub.png", 0.72), screen=True, pop=164.8, show=(164.8, 171.4)),
    dict(src=f"{C}/num1.png", x=960, y=470, w=nat(f"{C}/num1.png", 0.68), screen=True, pop=165.7, show=(165.7, 171.4)),
    dict(src=f"{C}/num1_sub.png", x=960, y=640, w=nat(f"{C}/num1_sub.png", 0.72), screen=True, pop=165.9, show=(165.9, 171.4)),
    dict(src=f"{C}/num2.png", x=1560, y=470, w=nat(f"{C}/num2.png", 0.68), screen=True, pop=166.8, show=(166.8, 171.4)),
    dict(src=f"{C}/num2_sub.png", x=1560, y=640, w=nat(f"{C}/num2_sub.png", 0.72), screen=True, pop=167.0, show=(167.0, 171.4)),
    dict(src=f"{C}/end_title.png", x=960, y=450, w=nat(f"{C}/end_title.png"), screen=True, pop=172.2, pop_dur=0.6),
    dict(src=f"{C}/end_sub.png", x=960, y=640, w=nat(f"{C}/end_sub.png"), screen=True, pop=172.8),
    dict(src=f"{C}/end_models.png", x=960, y=730, w=nat(f"{C}/end_models.png", 0.6), screen=True, pop=173.2),
]


def k(t, xy, zoom, dx=0, dy=0, **kw):
    return dict(t=t, x=xy[0] + dx, y=xy[1] + dy, zoom=zoom, **kw)


CAMERA = [
    # intro -> the photo -> the Camry, working
    k(0.0, (0, 0), 1.0), k(9.4, (0, 0), 1.0),
    k(11.0, (1150, 0), 0.5),
    k(12.4, PHOTO, 1.0, dy=-30), k(14.6, PHOTO, 1.06, dy=-25, ease="linear"),
    k(16.2, PROMISE, 1.38, dy=30), k(21.4, PROMISE, 1.48, dy=30, ease="linear"),
    # verified, and it works
    k(23.0, CHECK, 1.05), k(25.6, CHECK, 1.09, ease="linear"),
    k(27.2, HELI, 1.3, dy=40), k(30.0, HELI, 1.36, dy=40, ease="linear"),
    k(31.4, B747, 1.3, dy=40), k(34.6, B747, 1.36, dy=40, ease="linear"),
    k(36.6, PROMISE, 1.7, dy=40, arc=0.15), k(40.0, PROMISE, 1.8, dy=40, ease="linear"),
    # SAM 3.1 on the photo, then up into SAM 3D and the greedy fill
    k(42.0, PHOTO, 1.05, dy=-30), k(48.6, PHOTO, 1.12, dy=-30, ease="linear"),
    k(50.2, FILL, 1.32, dy=40), k(64.4, FILL, 1.45, dy=40, ease="linear"),
    # Spark and Glimmer edit it into a car
    k(66.0, COLLAB, 1.0, dx=420, dy=-20), k(82.0, COLLAB, 1.05, dx=420, dy=-20, ease="linear"),
    # the build, the manual, the parts
    k(83.8, BUILD, 1.32, dy=40), k(98.6, BUILD, 1.45, dy=40, ease="linear"),
    k(100.2, BOOK, 1.12), k(110.4, BOOK, 1.18, ease="linear"),
    k(111.8, PARTS, 0.95), k(114.4, PARTS, 0.98, ease="linear"),
    # the machines, superimposed (a still frame, a slow push)
    k(117.0, GROUP, 0.6, dy=-60, arc=0.35), k(128.4, GROUP, 0.65, dy=-60, ease="linear"),
    # why it matters
    k(130.4, MFG, 0.95, dy=-820, arc=0.2), k(133.6, MFG, 0.98, dy=-800, ease="linear"),
    k(135.4, MFG, 0.72, dy=250), k(145.4, MFG, 0.76, dy=250, ease="linear"),
    k(147.0, MFG, 0.95, dy=1250), k(151.4, MFG, 0.98, dy=1250, ease="linear"),
    k(153.0, MFG, 0.9, dy=2100), k(158.4, MFG, 0.93, dy=2100, ease="linear"),
    # everything at once, then the numbers and the end card over it
    k(163.6, (8000, 200), 0.098), k(180.0, (8000, 200), 0.104, ease="linear"),
]
