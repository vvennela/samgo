"""The launch film's layout and camera (scripts/film.py): one canvas, one camera, no cuts.

World coordinates are pixels at zoom 1; a camera keyframe's (x, y) is the world point at the centre of the frame.
The camera follows one smooth curve through the keyframes (SMOOTH); placeholder pacing until the narration is in.

 y=-1350:  SAM 3D shape -> resolution dial (10x) -> greedy fill (slab) -> Spark+Glimmer -> paint -> doors -> steps -> parts
 y=0:      intro -> photo (SAM 3.1) -> the Camry, working -> any object (montage)
 y=+1320:                             helicopter -> 747
 then the held-out machines replacing each other, a pull-back over everything, the numbers, the end card.
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
BG_IMAGE = f"{FILM}/bg.png"
SMOOTH = True
T0 = 0.0
DURATION = 180.0


def nat(path, k=1.0):
    return Image.open(path).width * k


def fit(path, cx, cy, content_w, content_h=None, **kw):
    """Place a transparent frame by its content: the drawn model's bounding box centred at (cx, cy), content_w wide."""
    if not Path(path).exists():          # not rendered yet: left out of previews
        return dict(src=path, x=cx, y=cy, w=content_w, show=(0, 0))
    im = Image.open(path)
    x0, y0, x1, y1 = im.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    k = content_w / (x1 - x0)
    if content_h:
        k = min(k, content_h / (y1 - y0))      # tall models fit the height instead
    return dict(src=path, x=cx - ((x0 + x1) / 2 - im.width / 2) * k, y=cy - ((y0 + y1) / 2 - im.height / 2) * k,
                w=im.width * k, **kw)


def card(name, x, y, k=1.0, **kw):
    p = f"{C}/{name}.png"
    return dict(src=p, x=x, y=y, w=nat(p, k), **kw)


def chip(name, x, y, at, until, k=0.82):
    return card(name, x, y, k, pop=at, show=(at, until), fade=0.35, shadow=True, radius=0.05)


def swap(paths, cx, cy, content_w, t0, each, fade=0.18, content_h=None):
    """A montage in one spot: each frame replaces the last."""
    out = []
    for i, p in enumerate(paths):
        if not Path(p).exists():
            continue
        out.append(fit(p, cx, cy, content_w, content_h, show=(t0 + i * each, t0 + (i + 1) * each + (99 if i == len(paths) - 1 else 0)),
                       fade=fade))
    return out


MANUAL = sorted(str(p) for p in Path(f"{FILM}/manual").glob("p-*.png"))
SCRIM = f"{FILM}/scrim.png"
if not Path(SCRIM).exists():
    Image.new("RGBA", (1920, 1080), BG + (225,)).save(SCRIM)

# ---------------------------------------------------------------- where things are
PHOTO = (2350, 0)
PROMISE = (4200, 0)
ANY = (6050, 0)
HELI, B747 = (4200, 1320), (6050, 1320)
SHAPE = (2350, -1350)
RES = (4250, -1350)
FILL = (6150, -1350)
WORK = (8150, -1350)
CHIPS_X = WORK[0] + 970
STEPS_AT_XY = (10350, -1350)
PARTS = (12150, -1350)
UNSEEN = (14100, -1350)

# ---------------------------------------------------------------- when
T_SAM31 = 44.0
T_SHAPE = 52.0           # the SAM 3D shape (the fill shot's first 3 s: the mesh alone)
T_RES = 57.0             # 10x whole (8 s), then the layers
T_FILL = 70.0            # the fill shot again from its start: mesh, then the greedy plates (slab at +12.5 s)
T_WORK = 85.5            # Spark + Glimmer time-lapse (12.5 s)
T_PAINT = T_WORK + 12.5
T_DOORS = T_PAINT + 4.0
T_STEPS = 109.5
T_PARTS = 123.0
T_UNSEEN = 130.5

ANY_OBJECTS = [f"{FILM}/s_{o}/00000.png" for o in
               ("bed_f2b3a71f", "chair_6dc3eff5", "duck_19f3599a", "boat_439c61b8", "sofa_5979a6cd")]
MACHINES = [f"{FILM}/b747/00000.png", f"{FILM}/helicopter/00000.png", f"{FILM}/cessna/00000.png",
            f"{FILM}/turbine/00000.png", f"{FILM}/s_fan/00000.png"]

ELEMENTS = [
    # intro and the promise
    dict(src=f"{FILM}/intro", x=0, y=0, w=1920, start=0, radius=0.012, shadow=True),
    dict(src=f"{STEPS}/1_photo.png", x=PHOTO[0], y=PHOTO[1], w=1500, radius=0.018, shadow=True),
    dict(src=f"{FILM}/sam31_car.png", x=PHOTO[0], y=PHOTO[1], w=1500, radius=0.018, show=(T_SAM31 + 0.3, T_SHAPE + 1),
         fade=0.5),
    dict(src=f"{FILM}/sam31_glass.png", x=PHOTO[0], y=PHOTO[1], w=1500, radius=0.018,
         show=(T_SAM31 + 3.3, T_SHAPE + 1), fade=0.5),
    card("lbl_sam31", PHOTO[0], -400, 0.9, show=(T_SAM31 + 0.3, T_SHAPE + 1), fade=0.45),
    dict(src=f"{FILM}/camry_move", x=PROMISE[0], y=PROMISE[1], w=1920, start=15.5),
    *swap(ANY_OBJECTS, ANY[0], ANY[1], 820, 22.4, 1.35, content_h=520),
    dict(src=f"{FILM}/helicopter", x=HELI[0], y=HELI[1], w=1800, start=29.0, loop=True),
    dict(src=f"{FILM}/b747", x=B747[0], y=B747[1], w=1800, start=33.0, loop=True),

    # SAM 3D's shape, the resolution dial, the greedy fill at a buildable size
    dict(src=f"{FILM}/fill", x=SHAPE[0], y=SHAPE[1], w=1920, start=T_SHAPE, show=(0, T_SHAPE + 6.5), fade=0.6),
    card("lbl_sam3d", SHAPE[0], SHAPE[1] - 520, 0.9, show=(T_SHAPE, T_SHAPE + 6.0), fade=0.45),
    fit(f"{FILM}/res_whole/00000.png", RES[0], RES[1], 1250, show=(0, 0)),     # (sizing only)
    dict(src=f"{FILM}/res_whole", x=RES[0], y=RES[1] - 10, w=1550, start=T_RES, show=(T_RES - 1.5, T_RES + 8.2),
         fade=0.5),
    dict(src=f"{FILM}/res_layers", x=RES[0], y=RES[1] - 10, w=1550, start=T_RES + 8.0, show=(T_RES + 8.0, T_FILL + 1),
         fade=0.5),
    card("cap_res", RES[0], RES[1] - 430, 0.8, pop=T_RES + 0.8),
    card("cap_res2", RES[0], RES[1] + 380, 0.8, pop=T_RES + 8.5),
    dict(src=f"{FILM}/fill", x=FILL[0], y=FILL[1], w=1920, start=T_FILL),
    card("cap_slab", FILL[0], FILL[1] + 520, 0.85, pop=T_FILL + 11.0, show=(T_FILL + 11.0, T_WORK + 1)),

    # Spark and Glimmer edit the slab into a car, then glass and paint, then the doors
    dict(src=f"{FILM}/collab", x=WORK[0], y=WORK[1], w=1920, start=T_WORK, show=(0, T_PAINT + 0.6), fade=0.6),
    dict(src=f"{FILM}/painted", x=WORK[0], y=WORK[1], w=1920, start=T_PAINT, show=(T_PAINT, T_DOORS + 0.5), fade=0.5),
    dict(src=f"{FILM}/doors", x=WORK[0], y=WORK[1], w=1920, start=T_DOORS, show=(T_DOORS, 999), fade=0.5),
    chip("chip_spark1", CHIPS_X, -1700, T_WORK + 1.4, T_WORK + 7.2),
    chip("chip_glimmer1", CHIPS_X, -1480, T_WORK + 3.0, T_WORK + 7.2),
    chip("chip_check1", CHIPS_X, -1255, T_WORK + 6.6, T_WORK + 7.2),
    chip("chip_glimmer2", CHIPS_X, -1480, T_WORK + 7.4, T_WORK + 12.3),
    chip("chip_check2", CHIPS_X, -1255, T_WORK + 10.9, T_WORK + 12.3),
    chip("chip_glass", CHIPS_X, -1480, T_PAINT + 0.2, T_PAINT + 3.8),
    chip("chip_doors", CHIPS_X, -1480, T_DOORS + 1.2, T_DOORS + 6.0),

    # the certified build, step by step, then the parts
    dict(src=f"{FILM}/steps", x=STEPS_AT_XY[0], y=STEPS_AT_XY[1], w=1920, start=T_STEPS),
    dict(src=MANUAL[57], x=PARTS[0], y=PARTS[1] + 90, w=720, radius=0.02, shadow=True, pop=T_PARTS - 0.6),
    card("cap_parts", PARTS[0], PARTS[1] - 470, 0.8, pop=T_PARTS + 0.4),

    # the held-out machines, replacing each other in one spot
    *swap(MACHINES, UNSEEN[0], UNSEEN[1] + 70, 1100, T_UNSEEN + 0.6, 2.3, content_h=560),
    card("cap_unseen", UNSEEN[0], UNSEEN[1] - 560, 0.85, pop=T_UNSEEN + 1.2),

    # over everything: today/next, the numbers, the end card (fixed on screen)
    dict(src=SCRIM, x=960, y=540, w=1920, screen=True, show=(151.8, 181), fade=1.2),
    card("cap_mfg3", 960, 540, 0.9, screen=True, pop=152.5, show=(152.5, 159.0)),
    dict(src=f"{C}/num0.png", x=360, y=470, w=nat(f"{C}/num0.png", 0.68), screen=True, pop=160.4, show=(160.4, 168.4)),
    dict(src=f"{C}/num0_sub.png", x=360, y=640, w=nat(f"{C}/num0_sub.png", 0.72), screen=True, pop=160.6, show=(160.6, 168.4)),
    dict(src=f"{C}/num1.png", x=960, y=470, w=nat(f"{C}/num1.png", 0.68), screen=True, pop=161.6, show=(161.6, 168.4)),
    dict(src=f"{C}/num1_sub.png", x=960, y=640, w=nat(f"{C}/num1_sub.png", 0.72), screen=True, pop=161.8, show=(161.8, 168.4)),
    dict(src=f"{C}/num2.png", x=1560, y=470, w=nat(f"{C}/num2.png", 0.68), screen=True, pop=162.8, show=(162.8, 168.4)),
    dict(src=f"{C}/num2_sub.png", x=1560, y=640, w=nat(f"{C}/num2_sub.png", 0.72), screen=True, pop=163.0, show=(163.0, 168.4)),
    card("end_title", 960, 450, 1.0, screen=True, pop=169.4, pop_dur=0.6),
    card("end_sub", 960, 640, 1.0, screen=True, pop=170.0),
    card("end_models", 960, 730, 0.6, screen=True, pop=170.4),
]
ELEMENTS = [e for e in ELEMENTS if e.get("show") != (0, 0)]


def k(t, xy, zoom, dx=0, dy=0):
    return dict(t=t, x=xy[0] + dx, y=xy[1] + dy, zoom=zoom)


CAMERA = [
    k(0.0, (0, 0), 1.0), k(9.2, (0, 0), 1.0),
    k(11.2, (1200, 0), 0.52),
    k(12.8, PHOTO, 1.0, dy=-30), k(14.4, PHOTO, 1.05, dy=-25),
    k(16.4, PROMISE, 1.38, dy=30), k(20.8, PROMISE, 1.48, dy=30),
    k(22.4, ANY, 1.2, dy=20), k(29.2, ANY, 1.28, dy=20),
    k(30.6, HELI, 1.3, dy=40), k(33.0, HELI, 1.36, dy=40),
    k(34.6, B747, 1.3, dy=40), k(37.6, B747, 1.36, dy=40),
    k(39.6, PROMISE, 1.62, dy=40), k(42.2, PROMISE, 1.72, dy=40),
    # the pipeline
    k(44.2, PHOTO, 1.05, dy=-30), k(50.4, PHOTO, 1.12, dy=-30),
    k(52.4, SHAPE, 1.3, dy=30), k(55.6, SHAPE, 1.38, dy=30),
    k(57.4, RES, 0.98, dy=-60), k(68.6, RES, 1.04, dy=-60),
    k(70.6, FILL, 1.3, dy=40), k(83.6, FILL, 1.42, dy=40),
    k(85.6, WORK, 1.0, dx=420, dy=-20), k(107.6, WORK, 1.06, dx=420, dy=-20),
    k(109.6, STEPS_AT_XY, 1.3, dy=30), k(121.0, STEPS_AT_XY, 1.4, dy=30),
    k(123.0, PARTS, 1.0, dy=-60), k(128.4, PARTS, 1.05, dy=-60),
    k(130.6, UNSEEN, 1.15, dy=-60), k(143.0, UNSEEN, 1.22, dy=-60),
    # everything at once
    k(150.0, (7000, -100), 0.115), k(180.0, (7000, -100), 0.122),
]
