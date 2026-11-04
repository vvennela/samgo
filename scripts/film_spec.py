"""The launch film's layout and camera (scripts/film.py): one canvas, one camera, no cuts.

World coordinates are pixels at zoom 1; a camera keyframe's (x, y) is the world point at the centre of the frame.
Times are seconds (placeholder pacing until the narration is recorded: re-time by moving keyframes and start times).

Rows on the canvas:  y = 0      intro -> photo -> the finished Camry            (the promise)
                     y = +1750  parts + checklist -> helicopter -> 747          (verified, and it works)
                     y = -1700  photo/SAM 3.1 -> SAM 3D + greedy fill -> Spark -> Glimmer + checker -> build
                                -> manual (stop motion) -> parts list            (how it works)
                     x ~ 18000  the machine wall                                (it generalizes)
                     x ~ 23500  known parts -> an assembly that holds -> CAD    (why it matters)
                     then a pull-back over everything, the numbers, the end card.
Assets: FILM (env FILM_ASSETS), made by scripts/film_tiles.py, film_cards.py, film_mesh.py and pdftoppm.
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
DOT = (216, 222, 230)
GRID = 64
T0 = 0.0
DURATION = 180.0


def nat(path, k=1.0):
    """World width for a card shown at its natural pixel size times k."""
    return Image.open(path).width * k


def label(name, x, y, show, k=1.0):
    p = f"{C}/{name}.png"
    return dict(src=p, x=x, y=y, w=nat(p, k), show=show, fade=0.5)


# the manual as stop motion: the cover, the 56 steps (each a little shorter than the one before), the last held
MANUAL = sorted(str(p) for p in Path(f"{FILM}/manual").glob("p-*.png"))
FLIP = f"{FILM}/flip"
if not Path(FLIP).exists():
    Path(FLIP).mkdir()
    for i, p in enumerate(MANUAL[:57]):
        os.symlink(p, f"{FLIP}/{i:03d}.png")
SCHEDULE, t, d = [0.0], 0.9, 0.5
for _ in range(56):
    SCHEDULE.append(t); t += d; d = max(0.09, d * 0.92)

# a scrim for the numbers over the final pull-back
SCRIM = f"{FILM}/scrim.png"
if not Path(SCRIM).exists():
    Image.new("RGBA", (1920, 1080), BG + (225,)).save(SCRIM)

ELEMENTS = [
    # ---- row 0: the promise
    dict(src=f"{FILM}/intro", x=0, y=0, w=1920, start=0, radius=0.012, shadow=True),
    dict(src=f"{STEPS}/1_photo.png", x=2750, y=0, w=1500, radius=0.018, shadow=True),
    dict(src=f"{FILM}/sam31_car.png", x=2750, y=0, w=1500, radius=0.018, show=(46.0, 53.5), fade=0.6),
    dict(src=f"{FILM}/sam31_glass.png", x=2750, y=0, w=1500, radius=0.018, show=(49.0, 53.5), fade=0.6),
    label("lbl_photo", 2750, -420, (11.5, 24)),
    label("lbl_sam31", 2750, -420, (46.0, 53.5)),
    dict(src=f"{FILM}/camry_move", x=4550, y=0, w=1920, start=15.0, loop=True),

    # ---- row +1750: verified, and it works
    label("lbl_verified", 3350, 1130, (24, 31)),
    dict(src=MANUAL[57], x=2750, y=1750, w=820, radius=0.02, shadow=True),
    dict(src=f"{C}/checklist.png", x=3950, y=1750, w=1250, shadow=True, radius=0.02),
    dict(src=f"{FILM}/helicopter", x=5550, y=1750, w=1800, start=30.0, loop=True),
    dict(src=f"{FILM}/b747", x=7300, y=1750, w=1800, start=33.0, loop=True),

    # ---- row -1700: how it works
    dict(src=f"{FILM}/fill", x=4550, y=-1700, w=1920, start=53.5),
    label("lbl_sam3d", 4550, -2230, (53.0, 69.0)),
    dict(src=f"{C}/spark", x=6450, y=-1700, w=1500, start=69.5, shadow=True, radius=0.014),
    label("lbl_spark", 6450, -2150, (68.0, 74.5)),
    dict(src=f"{C}/terminal", x=8400, y=-1700, w=1750, start=74.5, radius=0.0, shadow=True),
    label("lbl_glimmer", 8400, -2250, (73.5, 82.5)),
    dict(src=f"{FILM}/camry", x=10450, y=-1700, w=1920, start=82.0),
    label("lbl_build", 10450, -2230, (82.0, 96.5)),
    dict(src=FLIP, x=12500, y=-1700, w=1450, start=97.0, schedule=SCHEDULE, radius=0.012, shadow=True),
    label("lbl_manual", 12500, -2290, (96.0, 108.0)),
    dict(src=MANUAL[57], x=14000, y=-1700, w=800, radius=0.02, shadow=True),
    dict(src=MANUAL[58], x=14900, y=-1700, w=800, radius=0.02, shadow=True),
    label("lbl_parts", 14450, -2370, (107.5, 113.0)),

    # ---- the machine wall
    dict(src=f"{FILM}/b747", x=16100, y=-1050, w=1800, start=113, loop=True),
    dict(src=f"{FILM}/helicopter", x=18000, y=-1050, w=1800, start=113, loop=True),
    dict(src=f"{FILM}/cessna", x=19900, y=-1050, w=1800, start=113, loop=True),
    dict(src=f"{FILM}/gt1", x=16100, y=900, w=1800, start=113, loop=True),
    dict(src=f"{FILM}/turbine", x=18000, y=900, w=1800, start=113, loop=True),
    dict(src=f"{FILM}/camry_move", x=19900, y=900, w=1800, start=113, loop=True),
    dict(src=f"{C}/cap_unseen.png", x=18000, y=-2100, w=nat(f"{C}/cap_unseen.png", 1.6)),

    # ---- why it matters
    dict(src=f"{C}/cap_mfg1.png", x=23600, y=-800, w=nat(f"{C}/cap_mfg1.png")),
    dict(src=MANUAL[57], x=22350, y=250, w=700, radius=0.02, shadow=True),
    dict(src=FLIP, x=23550, y=250, w=1150, start=141.5, schedule=SCHEDULE, radius=0.012, shadow=True),
    dict(src=f"{FILM}/camry", x=25050, y=250, w=1600, start=141.5),
    dict(src=f"{C}/cap_mfg2.png", x=23600, y=1250, w=nat(f"{C}/cap_mfg2.png")),
    dict(src=f"{C}/cap_mfg3.png", x=23600, y=2150, w=nat(f"{C}/cap_mfg3.png")),

    # ---- over everything: the numbers, the end card (fixed on screen)
    dict(src=SCRIM, x=960, y=540, w=1920, screen=True, show=(168.5, 181), fade=1.2),
    dict(src=f"{C}/num0.png", x=400, y=470, w=nat(f"{C}/num0.png", 0.8), screen=True, show=(169.5, 175.5)),
    dict(src=f"{C}/num0_sub.png", x=400, y=610, w=nat(f"{C}/num0_sub.png", 0.62), screen=True, show=(169.5, 175.5)),
    dict(src=f"{C}/num1.png", x=960, y=470, w=nat(f"{C}/num1.png", 0.8), screen=True, show=(170.6, 175.5)),
    dict(src=f"{C}/num1_sub.png", x=960, y=610, w=nat(f"{C}/num1_sub.png", 0.62), screen=True, show=(170.6, 175.5)),
    dict(src=f"{C}/num2.png", x=1520, y=470, w=nat(f"{C}/num2.png", 0.8), screen=True, show=(171.7, 175.5)),
    dict(src=f"{C}/num2_sub.png", x=1520, y=610, w=nat(f"{C}/num2_sub.png", 0.62), screen=True, show=(171.7, 175.5)),
    dict(src=f"{C}/end_title.png", x=960, y=450, w=nat(f"{C}/end_title.png", 1.0), screen=True, show=(176.0, 181)),
    dict(src=f"{C}/end_sub.png", x=960, y=640, w=nat(f"{C}/end_sub.png"), screen=True, show=(176.6, 181)),
    dict(src=f"{C}/end_models.png", x=960, y=730, w=nat(f"{C}/end_models.png"), screen=True, show=(177.0, 181)),
]

CAMERA = [
    # intro, then the promise
    dict(t=0.0, x=0, y=0, zoom=1.0),
    dict(t=9.4, x=0, y=0, zoom=1.0),
    dict(t=11.8, x=1250, y=-60, zoom=0.42),
    dict(t=13.8, x=2750, y=-60, zoom=1.0),
    dict(t=16.0, x=2780, y=-50, zoom=1.07, ease="linear"),
    dict(t=18.0, x=4550, y=40, zoom=1.35, arc=0.2),
    dict(t=23.5, x=4550, y=40, zoom=1.45, ease="linear"),
    # verified, and it works
    dict(t=26.0, x=3350, y=1700, zoom=0.72),
    dict(t=30.0, x=3380, y=1720, zoom=0.78, ease="linear"),
    dict(t=32.0, x=5550, y=1790, zoom=1.2),
    dict(t=34.5, x=5560, y=1790, zoom=1.26, ease="linear"),
    dict(t=36.5, x=7300, y=1790, zoom=1.2),
    dict(t=39.0, x=7310, y=1790, zoom=1.26, ease="linear"),
    dict(t=41.5, x=4550, y=40, zoom=1.4, arc=0.35),
    dict(t=44.5, x=4550, y=40, zoom=1.5, ease="linear"),
    # how it works
    dict(t=46.5, x=2750, y=-80, zoom=1.0),
    dict(t=52.5, x=2780, y=-70, zoom=1.1, ease="linear"),
    dict(t=54.5, x=4550, y=-1660, zoom=1.3),
    dict(t=68.5, x=4550, y=-1660, zoom=1.42, ease="linear"),
    dict(t=70.5, x=6450, y=-1760, zoom=1.15),
    dict(t=73.0, x=6460, y=-1760, zoom=1.2, ease="linear"),
    dict(t=75.0, x=8400, y=-1800, zoom=0.98),
    dict(t=82.0, x=8400, y=-1800, zoom=1.0, ease="linear"),
    dict(t=84.0, x=10450, y=-1660, zoom=1.3),
    dict(t=96.0, x=10450, y=-1660, zoom=1.45, ease="linear"),
    # the manual, then the parts
    dict(t=98.0, x=12500, y=-1760, zoom=1.12),
    dict(t=107.0, x=12500, y=-1760, zoom=1.18, ease="linear"),
    dict(t=109.5, x=14450, y=-1800, zoom=0.8),
    dict(t=112.5, x=14450, y=-1800, zoom=0.84, ease="linear"),
    # the machine wall
    dict(t=116.0, x=18000, y=-450, zoom=0.34, arc=0.3),
    dict(t=119.0, x=18000, y=-450, zoom=0.36, ease="linear"),
    dict(t=121.5, x=16100, y=-1010, zoom=1.0),
    dict(t=124.5, x=18000, y=-1010, zoom=1.0),
    dict(t=127.5, x=19900, y=-1010, zoom=1.0),
    dict(t=130.5, x=16100, y=940, zoom=1.0, arc=0.15),
    dict(t=133.5, x=18000, y=940, zoom=1.0),
    dict(t=136.5, x=19900, y=940, zoom=1.0),
    dict(t=139.0, x=19000, y=-200, zoom=0.45),
    # why it matters
    dict(t=141.5, x=23600, y=-800, zoom=0.95, arc=0.2),
    dict(t=144.0, x=23600, y=-760, zoom=1.0, ease="linear"),
    dict(t=146.5, x=23700, y=250, zoom=0.56),
    dict(t=154.0, x=23700, y=250, zoom=0.6, ease="linear"),
    dict(t=156.5, x=23600, y=1250, zoom=0.9),
    dict(t=159.5, x=23600, y=1260, zoom=0.94, ease="linear"),
    dict(t=161.5, x=23600, y=2150, zoom=0.85),
    dict(t=164.5, x=23600, y=2150, zoom=0.9, ease="linear"),
    # everything at once, then the numbers and the end card over it
    dict(t=169.5, x=12600, y=200, zoom=0.066),
    dict(t=180.0, x=12600, y=200, zoom=0.07, ease="linear"),
]
