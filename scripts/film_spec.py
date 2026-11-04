"""The launch film: one centred shot at a time, changed with fades only (no camera moves, no slides, no pops).

Every element is fixed on screen (1920x1080 frame pixels). Models are placed by their measured extent, so each one is
centred and the same size whatever its render. Placeholder timing until the narration is recorded: each beat is a
start time below; moving one moves its shot.
(The earlier canvas-and-camera layout is kept in scripts/film_spec_canvas.py.)
"""
import json
import os
from pathlib import Path

from PIL import Image

FILM = os.environ.get("FILM_ASSETS", "/private/tmp/claude-501/-Users-vishnuv-Documents-Projects-samgo/"
                      "8e62d17f-cb50-41fb-8609-cb7914a1e9e1/scratchpad/film")
V5, V6, C = f"{FILM}/v5", f"{FILM}/v6", f"{FILM}/cards"
STEPS_DIR = "/Users/vishnuv/Documents/Projects/samgo/runs/video/steps"

W, H, FPS = 1920, 1080, 30
BG = (238, 241, 245)
BG_IMAGE = f"{FILM}/bg.png"
T0 = 0.0
FADE = 0.45
CAMERA = [dict(t=0, x=0, y=0, zoom=1.0)]       # unused: everything is on screen


def first(seq, k=0):
    p = Path(seq)
    if p.is_dir():
        fs = sorted(p.glob("*.png"))
        return fs[min(k, len(fs) - 1)]
    return p


def model(src, show, cx=960, cy=560, cw=1150, ch=640, k=0, fade=FADE, **kw):
    """A transparent render centred on screen by the model's own extent (frame k of a sequence)."""
    im = Image.open(first(src, k))
    x0, y0, x1, y1 = im.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    s = min(cw / (x1 - x0), ch / (y1 - y0))
    return dict(src=str(src), screen=True, w=im.width * s, show=show, fade=fade,
                x=cx - ((x0 + x1) / 2 - im.width / 2) * s, y=cy - ((y0 + y1) / 2 - im.height / 2) * s, **kw)


def card(name, show, x=960, y=540, k=1.0, fade=FADE, **kw):
    p = f"{C}/{name}.png"
    return dict(src=p, screen=True, x=x, y=y, w=Image.open(p).width * k, show=show, fade=fade, **kw)


def image(src, show, w, x=960, y=540, fade=FADE, **kw):
    return dict(src=str(src), screen=True, x=x, y=y, w=w, show=show, fade=fade, **kw)


def chips(rows, x=1500):
    """Speech cards on the right third: (name, y, t_in, t_out)."""
    out = []
    for name, y, a, b in rows:
        p = f"{C}/{name}.png"
        out.append(dict(src=p, screen=True, x=x, y=y, w=Image.open(p).width * 0.66, show=(a, b), fade=0.3,
                        shadow=True, radius=0.05))
    return out


MANUAL = sorted(str(p) for p in Path(f"{FILM}/manual").glob("p-*.png"))
FLIP = f"{FILM}/flip"                                  # the cover and the 56 step pages, in order
SCHEDULE, t, d = [0.0], 0.8, 0.32
for _ in range(56):
    SCHEDULE.append(round(t, 3)); t += d; d = max(0.075, d * 0.93)
FLIP_LEN = t + 0.9

# ---------------------------------------------------------------- the beats (start times)
T_PHOTO = 9.6
T_HERO = 12.6
T_ANY = 19.6
T_SAM31 = 26.4
T_MESH = 39.4
T_FULL = 43.6
T_SPLIT = 47.4                    # the layers come apart, 5x speed (360 frames -> 2.4 s)
T_DOWN = 51.0                     # 144,753 pieces -> 293 pieces
T_OVER = 56.0                     # the 293-piece model over SAM 3D's mesh
T_WORK = 62.4                     # Spark + Glimmer on the slab (12.5 s), paint (4 s), doors (6 s)
T_PAINT = T_WORK + 12.5
T_DOORS = T_PAINT + 4.0
T_SKILL = T_DOORS + 6.4
T_MANUAL = T_SKILL + 6.5
T_PARTS = T_MANUAL + FLIP_LEN
T_UNSEEN = T_PARTS + 5.0
T_CLOSE = T_UNSEEN + 4 * 5.6 + 0.4
DURATION = T_CLOSE + 17.0

ANY = [f"{V6}/{n}/00000.png" for n in ("bed", "chair", "skate", "tv", "laptop")]
ORBITS = [f"{V6}/o_gt1", f"{V6}/o_747", f"{V6}/o_heli", f"{V6}/o_cessna"]   # 6 s half-orbits

# the intro's words (scripts/film_words.py), with whisper's mishearings fixed
FIX = {"drop,": "draw,", "drop": "draw", "Sam": "SAMgo", "Go?": ""}
WORDS = json.loads(Path(f"{FILM}/words_intro.json").read_text()) if Path(f"{FILM}/words_intro.json").exists() else []
WORDS = [dict(w, w=FIX.get(w["w"], w["w"])) for w in WORDS]
WORDS = [w for w in WORDS if w["w"]]
_at = {w["w"].strip(",.?").lower(): w for w in WORDS}
T_DRAW = _at.get("models", {"t0": 0.6})["t0"]              # the drawing appears as "models can draw" is said
T_BUT = _at.get("but", {"t0": 2.4})["t0"]                  # and gives way to you on "but can they build..."

ELEMENTS = [
    # intro: "We know models can draw" over Muse Image's drawing, "but can they build..." on you, then the title
    image(f"{FILM}/intro", (0, T_PHOTO + 0.3), 1920, start=0, fade=0.0),
    image("/Users/vishnuv/Documents/Projects/samgo/runs/video/muse_image/lego_sedan_a.png", (T_DRAW, T_BUT), 1620,
          y=520, fade=0.25),
    card("lbl_drawn", (T_DRAW + 0.2, T_BUT), y=1010, k=0.9, fade=0.25),
    # one photo -> the LEGO Camry, working
    image(f"{STEPS_DIR}/1_photo.png", (T_PHOTO, T_HERO + 0.3), 1500, radius=0.018, shadow=True),
    model(f"{V5}/hero", (T_HERO, T_ANY + 0.2), start=T_HERO, cw=1250, ch=640),
    # any object
    card("cap_any", (T_ANY, T_SAM31 + 0.2), y=150),
    *[model(p, (T_ANY + 1.3 * i, T_ANY + 1.3 * (i + 1) + (0.25 if i < 4 else 0.2 + T_SAM31 - T_ANY - 6.5)),
            cw=900, ch=560, cy=590, fade=0.25) for i, p in enumerate(ANY)],

    # SAM 3.1: the car and its glass
    card("lbl_sam31", (T_SAM31, T_MESH + 0.2), y=170, k=0.9),
    image(f"{STEPS_DIR}/1_photo.png", (T_SAM31, T_MESH + 0.2), 1500, y=580, radius=0.018, shadow=True),
    image(f"{FILM}/sam31_car.png", (T_SAM31 + 1.6, T_MESH + 0.2), 1500, y=580, radius=0.018),
    image(f"{FILM}/sam31_glass.png", (T_SAM31 + 4.4, T_MESH + 0.2), 1500, y=580, radius=0.018),
    # SAM 3D: one photo to a 3D shape
    card("lbl_sam3d", (T_MESH, T_FULL + 0.2), y=170, k=0.9),
    model(f"{V5}/fill", (T_MESH, T_FULL + 0.2), start=T_MESH, frames=(0, 89), cw=1150, ch=600, cy=580),
    # full scale: 144,753 pieces, every piece connected; then it comes apart layer by layer
    card("cap_res", (T_FULL, T_DOWN + 0.2), y=150, k=0.8),
    model(f"{FILM}/res_layers", (T_FULL, T_DOWN + 0.2), start=T_SPLIT, rate=5, k=359, cw=1200, ch=600, cy=560, fade=0.4),
    card("cap_valid", (T_FULL + 1.2, T_DOWN + 0.2), y=960, k=0.85),
    # scaled down: 144,753 -> 293 pieces
    model(f"{FILM}/res_layers/00001.png", (T_DOWN, T_OVER + 0.2), cx=500, cy=560, cw=640, ch=420),
    card("arrow", (T_DOWN + 0.6, T_OVER + 0.2), x=960, y=560, k=0.55),
    model(f"{V6}/slab/00000.png", (T_DOWN + 1.0, T_OVER + 0.2), cx=1420, cy=560, cw=640, ch=420),
    image(f"{C}/piece_count", (T_DOWN + 0.2, T_OVER + 0.2), 640, x=960, y=850, start=T_DOWN + 0.8),
    # the 293-piece model over SAM 3D's mesh: same shape, real LEGO
    model(f"{V6}/mesh/00000.png", (T_OVER, T_WORK + 0.2), cw=1250, ch=650, cy=560),
    model(f"{V6}/slab/00000.png", (T_OVER + 1.4, T_WORK + 0.2), cw=1250, ch=650, cy=560, alpha=0.62),
    card("cap_shape", (T_OVER + 2.0, T_WORK + 0.2), y=960, k=0.8),

    # Muse Spark and Muse Glimmer make it a car: wheels and axles, glass and paint, doors
    model(f"{V5}/collab", (T_WORK, T_PAINT + 0.35), start=T_WORK, cx=700, cw=1000, ch=600, cy=580),
    model(f"{V5}/painted", (T_PAINT, T_DOORS + 0.35), start=T_PAINT, cx=700, cw=1000, ch=600, cy=580, fade=0.35),
    model(f"{V5}/doors", (T_DOORS, T_SKILL + 0.2), start=T_DOORS, cx=700, cw=1000, ch=600, cy=580, fade=0.35),
    *chips([("chip_spark1", 330, T_WORK + 1.4, T_WORK + 7.2), ("chip_glimmer1", 520, T_WORK + 3.0, T_WORK + 7.2),
            ("chip_check1", 720, T_WORK + 6.4, T_WORK + 7.2), ("chip_glimmer2", 520, T_WORK + 7.5, T_WORK + 12.3),
            ("chip_check2", 720, T_WORK + 10.8, T_WORK + 12.3), ("chip_glass", 520, T_PAINT + 0.2, T_PAINT + 3.9),
            ("chip_spark2", 330, T_DOORS - 1.2, T_DOORS + 5.9), ("chip_doors", 520, T_DOORS + 1.4, T_DOORS + 5.9)]),

    # what Glimmer learned: valid hinges and axles, which it could not do before
    card("cap_skill", (T_SKILL, T_MANUAL + 0.2), y=290, k=0.78),
    image(f"{C}/skill_count", (T_SKILL + 0.6, T_MANUAL + 0.2), 1500, x=1010, y=580, start=T_SKILL + 0.8),

    # the build manual, flipped through like stop motion: the only building in the film
    card("cap_manual", (T_MANUAL, T_PARTS + 0.2), y=95, k=0.8),
    image(FLIP, (T_MANUAL, T_PARTS + 0.2), 1300, y=580, start=T_MANUAL, schedule=SCHEDULE, fade=0.3),
    # real parts
    card("cap_parts", (T_PARTS, T_UNSEEN + 0.2), y=120, k=0.8),
    image(MANUAL[57], (T_PARTS, T_UNSEEN + 0.2), 640, y=600),

    # the montage: each machine turning, one after the next (the supercar is a car, a trained family: the
    # "never trained on" line shows only over the 747, the helicopter and the Cessna)
    *[model(p, (T_UNSEEN + 5.6 * i, T_UNSEEN + 5.6 * (i + 1) + 0.3), start=T_UNSEEN + 5.6 * i - 0.2, k=40,
            cw=1150, ch=600, cy=600, fade=0.4) for i, p in enumerate(ORBITS) if Path(p).exists()],
    card("cap_unseen", (T_UNSEEN + 5.6, T_CLOSE + 0.2), y=140, k=0.85),

    # the close
    card("cap_mfg3", (T_CLOSE + 0.2, T_CLOSE + 5.4), k=0.9),
    *[card(f"num{i}", (T_CLOSE + 5.8 + 0.5 * i, T_CLOSE + 11.4), x=360 + 600 * i, y=470, k=0.68) for i in range(3)],
    *[card(f"num{i}_sub", (T_CLOSE + 5.9 + 0.5 * i, T_CLOSE + 11.4), x=360 + 600 * i, y=640, k=0.72)
      for i in range(3)],
    card("end_title", (T_CLOSE + 11.8, 999), y=450),
    card("end_sub", (T_CLOSE + 12.2, 999), y=640),
    card("end_models", (T_CLOSE + 12.5, 999), y=730, k=0.6),
]


# ---------------------------------------------------------------- the stud dissolves (scripts/film_transitions.py)
CUTS = [T_HERO, T_SAM31, T_DOWN, T_MANUAL, T_UNSEEN]
if not os.environ.get("FILM_NO_TRANSITIONS"):
    for i, t in enumerate(CUTS):
        d = Path(f"{FILM}/trans/{i}")
        if d.exists() and any(d.iterdir()):
            ELEMENTS.append(dict(src=str(d), screen=True, x=960, y=540, w=1920, start=t - 0.4, show=(t - 0.4, t + 0.4),
                                 fade=0.0))

# captions last, over everything (the title line is the title card's, so it is not captioned)
ELEMENTS.append(dict(words=[w for w in WORDS if w["t0"] < _at.get("introducing", {"t0": 99})["t0"]], y=1000, size=44))
