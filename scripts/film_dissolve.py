"""The stud dissolve: one picture turns into LEGO studs, the studs flip to the next picture's studs, and resolve into it.

The picture is snapped to a grid of 1x1 LEGO cells (each the average colour of its patch, drawn as a plate seen from
above: a stud with a highlight and a soft shadow). Cells take over in a sweeping wave with a little randomness, so
the change reads as bricks clicking in, not a crossfade.

    python scripts/film_dissolve.py a.png b.png out_dir [frames=24] [cell=30]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080


def tile(cell: int) -> tuple[np.ndarray, np.ndarray]:
    """Shading for one cell: a multiplier (edges darker, stud shadow) and an additive highlight, both cell x cell."""
    s = 4 * cell                                          # draw large, then shrink (antialiased)
    mul = Image.new("L", (s, s), 255)
    d = ImageDraw.Draw(mul)
    d.rectangle([0, 0, s - 1, s - 1], outline=170, width=max(2, s // 20))          # the seam between plates
    r = s * 0.31
    c = s / 2
    shadow = Image.new("L", (s, s), 0)
    ImageDraw.Draw(shadow).ellipse([c - r + s * 0.05, c - r + s * 0.08, c + r + s * 0.05, c + r + s * 0.08], fill=80)
    shadow = shadow.filter(ImageFilter.GaussianBlur(s * 0.04))
    mul = Image.fromarray(np.clip(np.asarray(mul, float) - np.asarray(shadow, float), 0, 255).astype(np.uint8))
    hi = Image.new("L", (s, s), 0)
    hd = ImageDraw.Draw(hi)
    hd.ellipse([c - r, c - r, c + r, c + r], outline=70, width=max(2, s // 18))     # the stud's rim
    hd.ellipse([c - r * 0.55, c - r * 0.75, c + r * 0.05, c - r * 0.2], fill=60)     # its highlight
    hi = hi.filter(ImageFilter.GaussianBlur(s * 0.015))
    mul = np.asarray(mul.resize((cell, cell), Image.LANCZOS), float) / 255
    hi = np.asarray(hi.resize((cell, cell), Image.LANCZOS), float)
    return mul, hi


def studs(img: np.ndarray, cell: int, mul: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The picture as a field of 1x1 plates seen from above."""
    gh, gw = H // cell + 1, W // cell + 1
    pad = np.pad(img, ((0, gh * cell - H), (0, gw * cell - W), (0, 0)), mode="edge")
    avg = pad.reshape(gh, cell, gw, cell, 3).mean(axis=(1, 3))
    big = np.repeat(np.repeat(avg, cell, 0), cell, 1)
    shade = np.tile(mul, (gh, gw))[..., None]
    light = np.tile(hi, (gh, gw))[..., None]
    out = np.clip(big * shade + light, 0, 255)
    return out[:H, :W]


def order(cell: int, seed: int = 7) -> np.ndarray:
    """When each cell flips, 0..1: a left-to-right sweep with a slight diagonal and some jitter."""
    gh, gw = H // cell + 1, W // cell + 1
    yy, xx = np.mgrid[0:gh, 0:gw]
    rng = np.random.default_rng(seed)
    o = 0.78 * xx / gw + 0.12 * yy / gh + 0.18 * rng.random((gh, gw))
    o = (o - o.min()) / (o.max() - o.min())
    return np.repeat(np.repeat(o, cell, 0), cell, 1)[:H, :W]


def frames(a: Image.Image, b: Image.Image, n: int = 24, cell: int = 30):
    A = np.asarray(a.convert("RGB").resize((W, H)), float)
    B = np.asarray(b.convert("RGB").resize((W, H)), float)
    mul, hi = tile(cell)
    SA, SB = studs(A, cell, mul, hi), studs(B, cell, mul, hi)
    o = order(cell)[..., None]
    for k in range(n):
        u = (k + 1) / (n + 1)
        if u < 0.35:                                     # the picture becomes studs, cell by cell
            v = u / 0.35
            yield np.where(o < v, SA, A)
        elif u < 0.65:                                   # the studs flip to the next picture's studs
            v = (u - 0.35) / 0.30
            yield np.where(o < v, SB, SA)
        else:                                            # and resolve into it
            v = (u - 0.65) / 0.35
            yield np.where(o < v, B, SB)


def main():
    a, b, out = Image.open(sys.argv[1]), Image.open(sys.argv[2]), Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    cell = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    out.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(frames(a, b, n, cell)):
        Image.fromarray(f.astype(np.uint8)).save(out / f"{k:05d}.png")
    print(n, "frames ->", out)


if __name__ == "__main__":
    main()
