"""The launch film's layout and camera (scripts/film.py): one canvas, one camera, no cuts, no zoom-outs.

Every move is a straight glide between neighbours at about the same scale; the camera frames each model by its
measured extent (frame_on). The Camry keeps one spot (CAR) through the pipeline: SAM 3D's mesh, held while the camera
visits the full-scale clips, then filled with plates, then Spark and Glimmer, paint, doors.

 y=-1350:  CAR (mesh -> plates -> Spark+Glimmer -> paint -> doors)   RES (full scale) / STEPS   PARTS   UNSEEN
 y=0:      intro   photo (SAM 3.1)   the Camry (hero)   any object
 y=+1300:                            747                helicopter
Placeholder pacing until the narration is in.
"""
import os
from pathlib import Path

from PIL import Image

FILM = os.environ.get("FILM_ASSETS", "/private/tmp/claude-501/-Users-vishnuv-Documents-Projects-samgo/"
                      "8e62d17f-cb50-41fb-8609-cb7914a1e9e1/scratchpad/film")
V5 = f"{FILM}/v5"
C = f"{FILM}/cards"
STEPS_DIR = "/Users/vishnuv/Documents/Projects/samgo/runs/video/steps"

W, H, FPS = 1920, 1080, 30
BG = (238, 241, 245)
BG_IMAGE = f"{FILM}/bg.png"
SMOOTH = True
T0 = 0.0
DURATION = 168.0          # trimmed to T_END + 18 below


def nat(path, k=1.0):
    return Image.open(path).width * k


def _bbox(path):
    im = Image.open(path)
    return im.size, im.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()


def fit(path, cx, cy, content_w, content_h=None, **kw):
    """Place a transparent frame by its content: the drawn model's bounding box centred at (cx, cy)."""
    if not Path(path).exists():
        return dict(src=path, x=cx, y=cy, w=content_w, show=(0, 0))
    (iw, ih), (x0, y0, x1, y1) = _bbox(path)
    k = content_w / (x1 - x0)
    if content_h:
        k = min(k, content_h / (y1 - y0))
    return dict(src=path, x=cx - ((x0 + x1) / 2 - iw / 2) * k, y=cy - ((y0 + y1) / 2 - ih / 2) * k, w=iw * k, **kw)


def frame_on(t, seq, at, tile_w=1920, frame=0, fill=0.62):
    """A camera keyframe centred on the model drawn in frame `frame` of the sequence placed at `at` (tile_w wide), so
    it spans `fill` of the screen (whichever of width or height is tighter)."""
    files = sorted(Path(seq).glob("*.png")) if Path(seq).is_dir() else [Path(seq)]
    if not files:
        return dict(t=t, x=at[0], y=at[1], zoom=1.3)
    (iw, ih), (x0, y0, x1, y1) = _bbox(files[min(frame, len(files) - 1)])
    k = tile_w / iw
    cx = at[0] + ((x0 + x1) / 2 - iw / 2) * k
    cy = at[1] + ((y0 + y1) / 2 - ih / 2) * k
    zoom = min(fill * W / ((x1 - x0) * k), fill * H / ((y1 - y0) * k))
    return dict(t=t, x=cx, y=cy, zoom=zoom)


def card(name, x, y, k=1.0, **kw):
    p = f"{C}/{name}.png"
    return dict(src=p, x=x, y=y, w=nat(p, k), **kw)


def chip(name, x, y, at, until, k=1.0):
    return card(name, x, y, k, pop=at, show=(at, until), fade=0.3, shadow=True, radius=0.05)


def swap(paths, cx, cy, content_w, t0, each, content_h=None, fade=0.15):
    """A montage in one spot: each model replaces the last."""
    out = [p for p in paths if Path(p).exists()]
    return [fit(p, cx, cy, content_w, content_h, fade=fade,
                show=(t0 + i * each, t0 + (i + 1) * each + (99 if i == len(out) - 1 else 0)))
            for i, p in enumerate(out)]


MANUAL = sorted(str(p) for p in Path(f"{FILM}/manual").glob("p-*.png"))
SCRIM = f"{FILM}/scrim.png"
if not Path(SCRIM).exists():
    Image.new("RGBA", (1920, 1080), BG + (225,)).save(SCRIM)

# ---------------------------------------------------------------- where things are
INTRO, PHOTO, HERO, ANY = (0, 0), (2250, 0), (4150, 0), (5750, 0)
B747, HELI = (4150, 1150), (5750, 1150)
CAR, RES, PARTS, UNSEEN = (2250, -1350), (4150, -1350), (6050, -1350), (7950, -1350)
STEPS_AT = RES                     # the full-scale clips are gone by then
CHIPS_X = CAR[0] + 980

# ---------------------------------------------------------------- when
T_HERO = 15.2
T_ANY = 23.4
T_HELI, T_747 = 30.6, 35.2
T_SAM31 = 41.4
T_MESH = 50.4                      # the fill shot's first 3 s: SAM 3D's mesh alone, then held
T_RES = 55.4                       # full scale (8 s), then its layers
T_FILL = 69.8                      # back at the car: the fill resumes at frame 90 (the plates)
T_WORK = 83.4                      # Spark + Glimmer (12.5 s), then paint (4 s), then doors (6 s)
T_PAINT = T_WORK + 12.5
T_DOORS = T_PAINT + 4.0
T_APART = T_DOORS + 6.0           # the same car comes apart in reverse certified order, then back together (11.5 s)
T_PARTS = T_APART + 13.0
T_UNSEEN = T_PARTS + 6.0
T_END = T_UNSEEN + 9.0

ANY_OBJECTS = [f"{FILM}/s_{o}/00000.png" for o in
               ("bed_f2b3a71f", "chair_6dc3eff5", "duck_19f3599a", "boat_439c61b8", "sofa_5979a6cd")]
MACHINES = [f"{FILM}/b747/00000.png", f"{V5}/s_heli/00000.png", f"{FILM}/cessna/00000.png"]

_hero = frame_on(T_HERO + 0.8, f"{V5}/hero", HERO, frame=0, fill=0.66)
_car = frame_on(T_MESH + 0.4, f"{V5}/fill", CAR, frame=0, fill=0.6)
_slab = frame_on(T_FILL, f"{V5}/fill", CAR, frame=449, fill=0.56)
_heli = frame_on(T_HELI, f"{V5}/heli", HELI, frame=30, fill=0.6)
_747 = frame_on(T_747, f"{FILM}/b747", B747, frame=0, fill=0.72)


# the time-lapse view: the car on the left two thirds, the chips on the right third, all inside the frame
_zw = _slab["zoom"] * 0.82
_work = dict(_slab, t=T_WORK, x=_slab["x"] + (960 - 700) / _zw, zoom=_zw)


def screen(view, sx, sy):
    """World point shown at screen pixel (sx, sy) in a view."""
    return view["x"] + (sx - 960) / view["zoom"], view["y"] + (sy - 540) / view["zoom"]


def on(view, name, sx, sy, px_w, **kw):
    """A card shown at screen (sx, sy), px_w pixels wide, in a view."""
    x, y = screen(view, sx, sy)
    p = f"{C}/{name}.png"
    return dict(src=p, x=x, y=y, w=px_w / view["zoom"], **kw)


def chip_on(view, name, sy, at, until):
    p = f"{C}/{name}.png"
    w = Image.open(p).width * 0.62
    x, y = screen(view, 1440, sy)
    return dict(src=p, x=x, y=y, w=w / view["zoom"], pop=at, show=(at, until), fade=0.3, shadow=True, radius=0.05)


_unseen = dict(t=T_UNSEEN, x=UNSEEN[0], y=UNSEEN[1], zoom=1.1)

ELEMENTS = [
    # intro, the photo, the Camry working
    dict(src=f"{FILM}/intro", x=INTRO[0], y=INTRO[1], w=1920, start=0, radius=0.012, shadow=True),
    dict(src=f"{STEPS_DIR}/1_photo.png", x=PHOTO[0], y=PHOTO[1], w=1500, radius=0.018, shadow=True),
    dict(src=f"{FILM}/sam31_car.png", x=PHOTO[0], y=PHOTO[1], w=1500, radius=0.018, show=(T_SAM31 + 0.3, T_MESH),
         fade=0.5),
    dict(src=f"{FILM}/sam31_glass.png", x=PHOTO[0], y=PHOTO[1], w=1500, radius=0.018, show=(T_SAM31 + 3.3, T_MESH),
         fade=0.5),
    card("lbl_sam31", PHOTO[0], -390, 0.9, show=(T_SAM31 + 0.3, T_MESH - 0.6), fade=0.4),
    dict(src=f"{V5}/hero", x=HERO[0], y=HERO[1], w=1920, start=T_HERO),
    *swap(ANY_OBJECTS, ANY[0], ANY[1], 760, T_ANY + 0.2, 1.3, content_h=470),
    dict(src=f"{FILM}/b747", x=B747[0], y=B747[1], w=1920, start=T_747 - 1, loop=True),
    dict(src=f"{V5}/heli", x=HELI[0], y=HELI[1], w=1920, start=T_HELI - 1),

    # the car's spot: SAM 3D's mesh (held), then the plates, then Spark and Glimmer, paint, doors
    dict(src=f"{V5}/fill", x=CAR[0], y=CAR[1], w=1920, start=T_MESH, frames=(0, 89), show=(0, T_FILL + 0.05), fade=0),
    dict(src=f"{V5}/fill", x=CAR[0], y=CAR[1], w=1920, start=T_FILL - 3.0, show=(T_FILL, T_WORK + 0.6), fade=0.6),
    on(_car, "lbl_sam3d", 960, 110, 760, show=(T_MESH, T_MESH + 4.4), fade=0.4),
    on(_slab, "cap_slab", 960, 985, 1150, pop=T_FILL + 10.6, show=(T_FILL + 10.6, T_WORK - 0.3)),
    dict(src=f"{V5}/collab", x=CAR[0], y=CAR[1], w=1920, start=T_WORK, show=(T_WORK, T_PAINT + 0.5), fade=0.5),
    dict(src=f"{V5}/painted", x=CAR[0], y=CAR[1], w=1920, start=T_PAINT, show=(T_PAINT, T_DOORS + 0.4), fade=0.4),
    dict(src=f"{V5}/doors", x=CAR[0], y=CAR[1], w=1920, start=T_DOORS, show=(T_DOORS, T_APART + 0.3), fade=0.3),
    dict(src=f"{V5}/apart", x=CAR[0], y=CAR[1], w=1920, start=T_APART, show=(T_APART, 999), fade=0.3),
    on(_slab, "cap_steps", 960, 985, 900, pop=T_APART + 6.0, show=(T_APART + 6.0, T_APART + 11.8)),
    chip_on(_work, "chip_spark1", 290, T_WORK + 1.4, T_WORK + 7.2),
    chip_on(_work, "chip_glimmer1", 470, T_WORK + 3.0, T_WORK + 7.2),
    chip_on(_work, "chip_check1", 660, T_WORK + 6.6, T_WORK + 7.2),
    chip_on(_work, "chip_glimmer2", 470, T_WORK + 7.5, T_WORK + 12.3),
    chip_on(_work, "chip_check2", 660, T_WORK + 10.9, T_WORK + 12.3),
    chip_on(_work, "chip_glass", 470, T_PAINT + 0.2, T_PAINT + 3.9),
    chip_on(_work, "chip_spark2", 290, T_PAINT + 2.6, T_DOORS + 5.8),
    chip_on(_work, "chip_doors", 470, T_DOORS + 1.4, T_DOORS + 5.8),

    # full scale, beside the car
    dict(src=f"{FILM}/res_whole", x=RES[0], y=RES[1] - 10, w=1550, start=T_RES, show=(T_RES - 1.0, T_RES + 8.2),
         fade=0.5),
    dict(src=f"{FILM}/res_layers", x=RES[0], y=RES[1] - 10, w=1550, start=T_RES + 8.0,
         show=(T_RES + 8.0, T_FILL + 0.8), fade=0.5),
    card("cap_res", RES[0], RES[1] - 430, 0.8, pop=T_RES + 0.6, show=(T_RES + 0.6, T_FILL + 0.8)),
    card("cap_valid", RES[0], RES[1] + 380, 0.85, pop=T_RES + 8.6, show=(T_RES + 8.6, T_RES + 11.6), fade=0.3),
    card("cap_res2", RES[0], RES[1] + 380, 0.8, pop=T_RES + 11.8, show=(T_RES + 11.8, T_FILL + 0.8)),

    # the parts; the held-out machines
    dict(src=MANUAL[57], x=PARTS[0], y=PARTS[1] + 90, w=700, radius=0.02, shadow=True, pop=T_PARTS - 0.4),
    card("cap_parts", PARTS[0], PARTS[1] - 460, 0.8, pop=T_PARTS + 0.4, show=(T_PARTS + 0.4, T_UNSEEN - 0.6)),
    *swap(MACHINES, UNSEEN[0], UNSEEN[1] + 60, 1050, T_UNSEEN + 0.3, 2.7, content_h=520),
    on(_unseen, "cap_unseen", 960, 120, 1100, pop=T_UNSEEN + 0.9, show=(T_UNSEEN + 0.9, T_END + 0.3)),

    # the close, over the last machine: today/next, the numbers, the end card
    dict(src=SCRIM, x=960, y=540, w=1920, screen=True, show=(T_END, 999), fade=1.0),
    card("cap_mfg3", 960, 540, 0.9, screen=True, pop=T_END + 0.6, show=(T_END + 0.6, T_END + 5.2)),
    *[dict(src=f"{C}/num{i}.png", x=360 + 600 * i, y=470, w=nat(f"{C}/num{i}.png", 0.68), screen=True,
           pop=T_END + 5.6 + 0.7 * i, show=(T_END + 5.6 + 0.7 * i, T_END + 11.4)) for i in range(3)],
    *[dict(src=f"{C}/num{i}_sub.png", x=360 + 600 * i, y=640, w=nat(f"{C}/num{i}_sub.png", 0.72), screen=True,
           pop=T_END + 5.8 + 0.7 * i, show=(T_END + 5.8 + 0.7 * i, T_END + 11.4)) for i in range(3)],
    card("end_title", 960, 450, 1.0, screen=True, pop=T_END + 12.0, pop_dur=0.6),
    card("end_sub", 960, 640, 1.0, screen=True, pop=T_END + 12.5),
    card("end_models", 960, 730, 0.6, screen=True, pop=T_END + 12.9),
]
ELEMENTS = [e for e in ELEMENTS if e.get("show") != (0, 0)]


def at(t, xy, zoom):
    return dict(t=t, x=xy[0], y=xy[1], zoom=zoom)


def hold(key, dt, push=1.04):
    """The same framing dt later, pushed in a touch (a slow drift instead of a dead stop)."""
    return dict(key, t=key["t"] + dt, zoom=key["zoom"] * push)


CAMERA = [
    at(0.0, INTRO, 1.0), at(9.0, INTRO, 1.0),
    at(10.8, PHOTO, 1.0), at(13.6, PHOTO, 1.04),
    _hero, hold(_hero, 6.4),
    at(T_ANY, ANY, 1.3), at(T_ANY + 6.6, ANY, 1.34),
    _heli, hold(_heli, 3.0),
    _747, hold(_747, 3.6),
    at(T_SAM31, PHOTO, 1.0), at(T_SAM31 + 7.6, PHOTO, 1.05),
    _car, hold(_car, 3.4),
    at(T_RES + 0.2, RES, 1.02), at(T_FILL - 1.8, RES, 1.06),
    _slab, hold(_slab, T_WORK - T_FILL - 0.8),
    _work, hold(_work, T_APART - T_WORK, 1.03),
    dict(_slab, t=T_APART + 1.0), hold(dict(_slab, t=T_APART + 1.0), 10.5),
    at(T_PARTS, PARTS, 1.0), at(T_PARTS + 4.4, PARTS, 1.03),
    _unseen, at(DURATION, UNSEEN, 1.16),
]

DURATION = T_END + 18.0
CAMERA[-1] = at(DURATION, UNSEEN, 1.16)
