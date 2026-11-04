"""2D cards for the launch film, in one type system (Futura on the canvas colours): stage labels, the verified
checklist, the Glimmer/checker terminal (typed, frame by frame), captions, numbers, the end card.

    PYTHONPATH=. python scripts/film_cards.py out_dir
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FUT = "/System/Library/Fonts/Supplemental/Futura.ttc"     # 0 Medium, 2 Bold
MONO = "/System/Library/Fonts/Menlo.ttc"
INK = (29, 35, 41)
MUTED = (110, 120, 132)
RED = (208, 16, 18)
GREEN = (22, 150, 80)
FPS = 30


def font(size, face=0, path=FUT):
    return ImageFont.truetype(path, size, index=face)


def text_png(path: Path, lines, size=72, face=0, color=INK, track=0.04, gap=0.35, align="center", pad=20):
    """Transparent PNG of one or more lines of tracked Futura."""
    f = font(size, face)
    def width(s):
        return sum(f.getlength(c) for c in s) + track * size * max(0, len(s) - 1)
    W = int(max(width(s) for s in lines)) + 2 * pad
    lh = int(size * (1 + gap))
    H = lh * len(lines) + 2 * pad
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, s in enumerate(lines):
        x = pad + ((W - 2 * pad - width(s)) / 2 if align == "center" else 0)
        y = pad + i * lh
        for c in s:
            d.text((x, y), c, font=f, fill=color + (255,))
            x += f.getlength(c) + track * size
    im.save(path)


def card(w, h, fill=(255, 255, 255), radius=28):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w - 1, h - 1], radius, fill=fill + (255,))
    return im


def tick(d, cx, cy, r, color):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (255,))
    d.line([(cx - r * 0.45, cy + r * 0.02), (cx - r * 0.1, cy + r * 0.38), (cx + r * 0.5, cy - r * 0.35)],
           fill=(255, 255, 255, 255), width=max(3, int(r * 0.22)), joint="curve")


def checklist(path: Path):
    rows = ["Every piece is a real LEGO part", "Every piece held by studs", "Every mechanism moves clear",
            "Stands at every step of the build"]
    W, H = 1500, 140 + 150 * len(rows)
    im = card(W, H)
    d = ImageDraw.Draw(im)
    f = font(60)
    for i, s in enumerate(rows):
        y = 100 + i * 150
        tick(d, 120, y + 32, 38, GREEN)
        d.text((200, y), s, font=f, fill=INK + (255,))
    im.save(path)


TERMINAL = [   # the real case: runs/audit/r9-sheets/r10.json, flap8_val_100006 (docs/feedback-example.md)
    ("dim", "Glimmer  >  round 1"),
    ("code", '{"op": "clear", "box": [4, 4, 29, 32, 0, 60]}'),
    ("code", '{"op": "add", "part": {"type": "clip_plate", "x": 2, "y": 29, "z": 11, ...}}'),
    ("code", '{"op": "add", "part": {"type": "clip_plate", "x": 2, "y": 32, "z": 11, ...}}'),
    ("gap", ""),
    ("red", "checker  x  connected: these hinge parts hold nothing - their studs"),
    ("red", "            grip no other part: your added (clip_plate 2x1 at (x=2, y=29))"),
    ("red", "            - put each on (or under) a part of the side it carries"),
    ("gap", ""),
    ("dim", "Glimmer  >  round 2"),
    ("code", '{"op": "clear", "box": [6, 6, 29, 32, 0, 60]}'),
    ("code", '{"op": "add", "part": {"type": "clip_plate", "x": 4, "y": 29, "z": 11, ...}}'),
    ("gap", ""),
    ("green", "checker  ok  passes every check"),
]


def terminal(out: Path, seconds=7.0):
    """The feedback moment, typed: Glimmer's answer, the checker's sentence in red, the fix, green."""
    out.mkdir(parents=True, exist_ok=True)
    W, H = 1900, 1000
    f = font(33, 0, MONO)
    colors = {"dim": (150, 160, 172), "code": (220, 226, 234), "red": (255, 92, 84), "green": (80, 220, 140)}
    chars = sum(len(s) for _, s in TERMINAL)
    n = int(seconds * FPS)
    for k in range(n):
        typed = int(chars * min(1.0, k / (0.85 * n)))
        im = card(W, H, (24, 28, 34), 30)
        d = ImageDraw.Draw(im)
        for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            d.ellipse([36 + i * 34, 30, 56 + i * 34, 50], fill=c + (255,))
        y, left = 100, typed
        for kind, s in TERMINAL:
            if kind == "gap":
                y += 26
                continue
            show = s[:max(0, left)]
            left -= len(s)
            if show:
                d.text((48, y), show, font=f, fill=colors[kind] + (255,))
            y += 54
            if left <= 0:
                if k % 30 < 18:   # caret
                    x = 48 + f.getlength(show)
                    d.rectangle([x + 4, y - 50, x + 22, y - 12], fill=(220, 226, 234, 255))
                break
        im.save(out / f"{k:05d}.png")


def spark(out: Path, target_npz: str, layout: dict, wheels_x=(4, 16), seconds=5.0):
    """Muse Spark's answer drawn on the car's side silhouette (SAM 3D's voxels on the stud grid), one mark at a
    time: the wheels, the doors, which end is the front."""
    import numpy as np
    out.mkdir(parents=True, exist_ok=True)
    g = np.load(target_npz)["target"] >= 0 if np.load(target_npz)["target"].min() < 0 else np.load(target_npz)["target"] > 0
    side = g.any(axis=1)                    # (x, z)
    nx, nz = side.shape
    W, H = 1900, 760
    sx = 70                                 # px per stud
    sz = sx * 0.4                           # px per plate (a plate is 0.4 of a stud wide)
    ox = (W - nx * sx) / 2
    oz = H - 120                            # ground line
    X = lambda x: ox + x * sx
    Z = lambda z: oz - z * sz
    f = font(40, 0)
    marks = []
    rr = 1.9 * sx                           # 30.4 mm tyre: ~3.8 studs across
    for wx, name in zip(wheels_x, ("WHEEL", "WHEEL")):
        marks.append(("wheel", (X(wx), Z(0) - rr * 0.55, rr), name))
    for dr, name in zip(sorted(layout["doors"], key=lambda d: d["x0"]), ("REAR DOOR", "FRONT DOOR")):
        marks.append(("door", (X(dr["x0"]), Z(dr["z1"]), X(dr["x1"]), Z(dr["z0"])), name))
    marks.append(("front", (X(nx) + 20, Z(nz * 0.45)), "FRONT"))
    n = int(seconds * FPS)
    for k in range(n):
        t = k / FPS
        im = card(W, H)
        d = ImageDraw.Draw(im)
        for x in range(nx):
            for z in range(nz):
                if side[x, z]:
                    d.rounded_rectangle([X(x) + 2, Z(z + 1) + 1, X(x + 1) - 2, Z(z) - 1], 3, fill=(205, 211, 219, 255))
        for i, (kind, geo, name) in enumerate(marks):
            u = (t - 0.5 - i * 0.65) / 0.35
            if u <= 0:
                continue
            a = int(255 * min(1, u))
            pop = 1 + 0.25 * max(0.0, 1 - u) if u < 1 else 1.0
            col = RED + (a,)
            if kind == "wheel":
                cx, cy, r = geo; r *= pop
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=9)
                d.text((cx, cy + r + 18), name, font=f, fill=col, anchor="mt")
            elif kind == "door":
                x0, y0, x1, y1 = geo
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                hw, hh = (x1 - x0) / 2 * pop, (y1 - y0) / 2 * pop
                d.rounded_rectangle([cx - hw + 6, cy - hh, cx + hw - 6, cy + hh], 10, outline=col, width=8)
                d.text((cx, cy - hh - 16), name, font=f, fill=col, anchor="mb")
            else:
                x, y = geo
                d.line([(x, y), (x + 90 * pop, y)], fill=col, width=9)
                d.polygon([(x + 90 * pop + 26, y), (x + 90 * pop - 4, y - 22), (x + 90 * pop - 4, y + 22)], fill=col)
                d.text((x + 10, y - 30), name, font=f, fill=col, anchor="lb")
        im.save(out / f"{k:05d}.png")


SPARK_C, GLIMMER_C = (124, 58, 237), (8, 102, 255)


def chip(path: Path, who: str, color, lines, mono=False, ok=False):
    """A small speech card: a coloured dot and the speaker's name, then its line(s)."""
    f_who = font(30, 2)
    f_txt = font(30, 0, MONO) if mono else font(40, 0)
    pad, gap = 34, 14
    width = max([f_who.getlength(who) + 70] + [f_txt.getlength(l) + (64 if ok else 0) for l in lines]) + 2 * pad
    lh = 46 if mono else 54
    height = pad + 40 + gap + lh * len(lines) + pad - 6
    im = card(int(width), int(height), (255, 255, 255), 26)
    d = ImageDraw.Draw(im)
    d.ellipse([pad, pad + 6, pad + 22, pad + 28], fill=color + (255,))
    x = pad + 36
    for c in who:
        d.text((x, pad), c, font=f_who, fill=color + (255,))
        x += f_who.getlength(c) + 3
    y = pad + 40 + gap
    for l in lines:
        tx = pad
        if ok:
            tick(d, pad + 20, y + 24, 20, GREEN)
            tx = pad + 56
        d.text((tx, y), l, font=f_txt, fill=INK + (255,))
        y += lh
    im.save(path)


def chips(out: Path):
    chip(out / "chip_spark1.png", "MUSE SPARK", SPARK_C, ["It's a car. The wheels go here, and here."])
    chip(out / "chip_glimmer1.png", "MUSE GLIMMER", GLIMMER_C,
         ["add technic_brick 1x4 at (2, 2..8, 2)", "add axle at x=4", "add wheel x2"], mono=True)
    chip(out / "chip_check1.png", "CHECKER", GREEN, ["Turns, held by studs"], ok=True)
    chip(out / "chip_glimmer2.png", "MUSE GLIMMER", GLIMMER_C,
         ["add technic_brick 1x4 at (14, 2..8, 2)", "add axle at x=16", "add wheel x2"], mono=True)
    chip(out / "chip_check2.png", "CHECKER", GREEN, ["Rolls, and stands on its wheels"], ok=True)
    chip(out / "chip_doors.png", "CHECKER", GREEN, ["Doors swing clear on their hinges"], ok=True)
    chip(out / "chip_spark2.png", "MUSE SPARK", SPARK_C, ["It has doors, hinged at the front."])
    chip(out / "chip_glass.png", "SAM 3.1 + MUSE SPARK", (90, 100, 112), ["Glass where the photo has it. Painted red."])


def arrow(path: Path, length=520, color=INK):
    im = Image.new("RGBA", (length + 40, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.line([(20, 60), (length - 30, 60)], fill=color + (255,), width=10)
    d.polygon([(length + 20, 60), (length - 50, 22), (length - 50, 98)], fill=color + (255,))
    im.save(path)


def stats(path: Path, rows):
    """Before/now rows: label, before, now."""
    f_l, f_n = font(46, 0), font(84, 0)
    W, rh = 1500, 190
    im = Image.new("RGBA", (W, rh * len(rows)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, (label, before, now) in enumerate(rows):
        y = i * rh
        d.text((0, y + 8), label, font=f_l, fill=MUTED + (255,))
        d.text((0, y + 70), before, font=f_n, fill=(170, 176, 186, 255))
        x = 30 + f_n.getlength(before)
        d.line([(x, y + 118), (x + 90, y + 118)], fill=INK + (255,), width=7)
        d.polygon([(x + 120, y + 118), (x + 86, y + 98), (x + 86, y + 138)], fill=INK + (255,))
        d.text((x + 150, y + 70), now, font=f_n, fill=GREEN + (255,))
    im.save(path)


def counters(out: Path):
    """Moving numbers: the skill stats counting up (1.2 s), and 144,753 rolling down to 293 (1.6 s)."""
    ease = lambda u: 1 - (1 - min(1.0, max(0.0, u))) ** 3
    d = out / "skill_count"; d.mkdir(exist_ok=True)
    n = 45
    for k in range(n):
        u = ease(k / 36)
        stats(d / f"{k:05d}.png", [("DOOR HINGES ON VEHICLES IT NEVER SAW", "0/10", f"{round(10 * u)}/10"),
                                   ("HINGING A PART ON BODIES IT NEVER SAW", "0/20", f"{round(15 * u)}/20")])
    d = out / "piece_count"; d.mkdir(exist_ok=True)
    for k in range(60):
        u = ease(k / 48)
        v = round(144753 * (293 / 144753) ** u)       # geometric: the count falls like a dial turning down
        text_png(d / f"{k:05d}.png", [f"{v:,} PIECES"], size=64, color=INK, track=0.08)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    counters(out)
    text_png(out / "lbl_drawn.png", ["DRAWN BY META MUSE IMAGE"], size=40, color=MUTED, track=0.14)
    arrow(out / "arrow.png")
    stats(out / "skill_stats.png", [("DOOR HINGES ON VEHICLES IT NEVER SAW", "0/10", "10/10"),
                                     ("HINGING A PART ON BODIES IT NEVER SAW", "0/20", "15/20")])
    text_png(out / "cap_skill.png", ["GLIMMER LEARNED TO BUILD WORKING HINGES AND AXLES"], size=60, track=0.08)
    text_png(out / "cap_skill2.png", ["ITS OWN ANSWER, THE CHECKER'S NOTE, THE FIX"], size=44, color=MUTED, track=0.1)
    text_png(out / "lbl_full.png", ["144,753 PIECES"], size=56, color=MUTED, track=0.1)
    text_png(out / "lbl_small.png", ["293 PIECES"], size=56, color=MUTED, track=0.1)
    text_png(out / "cap_shape.png", ["SAMGO BUILDS THE WHOLE BODY IN REAL LEGO, AND KEEPS ITS SHAPE"], size=56, track=0.08)
    text_png(out / "cap_manual.png", ["THE BUILD MANUAL  ·  56 STEPS, EACH ONE CHECKED TO HOLD"], size=52, color=MUTED, track=0.1)
    text_png(out / "cap_any.png", ["ANY OBJECT"], size=60, color=MUTED, track=0.14)
    text_png(out / "cap_works.png", ["AND IT WORKS"], size=60, color=MUTED, track=0.14)
    labels = {
        "lbl_photo": "ONE PHOTO", "lbl_sam31": "SAM 3.1  ·  FINDS THE CAR AND ITS GLASS",
        "lbl_sam3d": "SAM 3D  ·  ONE PHOTO TO A 3D SHAPE",
        "lbl_spark": "MUSE SPARK  ·  KNOWS WHAT MOVES", "lbl_glimmer": "MUSE GLIMMER  ·  BUILDS IT, THE CHECKER CHECKS",
        "lbl_build": "THE BUILD", "lbl_manual": "THE MANUAL  ·  56 STEPS, EACH ONE CHECKED",
        "lbl_parts": "320 REAL PARTS, READY TO ORDER", "lbl_verified": "VERIFIED, NOT GENERATED",
    }
    for name, s in labels.items():
        text_png(out / f"{name}.png", [s], size=54, color=MUTED, track=0.12)
    text_png(out / "cap_res.png", ["FULL SCALE  ·  144,753 PIECES"], size=96, track=0.1)
    text_png(out / "cap_valid.png", ["EVERY PIECE CONNECTED"], size=64, color=MUTED, track=0.1)
    text_png(out / "cap_layers.png", ["LAYER BY LAYER"], size=64, color=MUTED, track=0.1)
    text_png(out / "cap_res2.png", ["SCALED DOWN TO SOMETHING YOU CAN BUILD"], size=64, color=MUTED, track=0.1)
    text_png(out / "cap_slab.png", ["REAL PLATES.  NO WHEELS, NO DOORS, NOTHING MOVES."], size=60, color=MUTED, track=0.1)
    text_png(out / "cap_steps.png", ["56 STEPS, EACH ONE CHECKED TO HOLD"], size=64, color=MUTED, track=0.1)
    text_png(out / "cap_parts.png", ["THIS IS A REAL BUILD, WITH REAL PARTS"], size=84, track=0.08)
    text_png(out / "cap_unseen.png", ["NONE OF THESE WERE IN THE TRAINING DATA"], size=84, track=0.1)
    text_png(out / "cap_mfg1.png", ["KNOWN PARTS"], size=150, track=0.12)
    text_png(out / "cap_mfg2.png", ["AN ASSEMBLY THAT HOLDS AT EVERY STEP"], size=84, track=0.08)
    text_png(out / "cap_mfg3.png", ["TODAY: LEGO.  NEXT: REAL CAD."], size=110, track=0.08)
    nums = [("50/50", ["HELD-OUT CARS", "BUILT"]), ("10/10", ["DOORS ON VEHICLES", "IT NEVER SAW"]),
            ("3×", ["A NEW SKILL IN ONE RUN", "8 TO 24 OF 30"])]
    for i, (big, small) in enumerate(nums):
        text_png(out / f"num{i}.png", [big], size=200, face=0, track=0.02)
        text_png(out / f"num{i}_sub.png", small, size=44, color=MUTED, track=0.1, gap=0.4)
    text_png(out / "end_title.png", ["SAMGO"], size=260, track=0.06)
    text_png(out / "end_sub.png", ["github.com/vvennela/samgo"], size=60, color=MUTED, track=0.04)
    text_png(out / "end_models.png", ["BUILT WITH  SAM 3.1  ·  SAM 3D  ·  MUSE SPARK  ·  MUSE GLIMMER"], size=48,
             color=MUTED, track=0.12)
    checklist(out / "checklist.png")
    terminal(out / "terminal")
    chips(out)
    if len(sys.argv) > 2:   # run dir of the pipeline build: Spark's layout on its voxel side view
        import json
        run = Path(sys.argv[2])
        summ = json.loads((run / "summary.json").read_text())
        spark(out / "spark", str(run / "target.npz"), summ["layout"], tuple(w[0] for w in summ["wheels"]))
    print("cards ->", out)


if __name__ == "__main__":
    main()
