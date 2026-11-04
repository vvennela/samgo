"""The build-complete bounce (as in the LEGO games): the finished model squashes a little, hops up stretching, lands
with a squash and settles with a small rebound, its contact shadow shrinking while it is in the air. Frames from a
transparent render (scripts/film_tiles.py) over the film's background.

    python scripts/film_bounce.py <transparent frames dir> bg.png out.mp4 [model width px=1250]
"""
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

FPS = 30
# (time s, lift px, x scale, y scale): squash, hop and stretch, apex, land and squash, rebound, settle
KEYS = [(0.00, 0, 1.00, 1.00), (0.30, 0, 1.00, 1.00), (0.42, 0, 1.05, 0.92), (0.58, 70, 0.97, 1.06),
        (0.74, 95, 1.00, 1.00), (0.92, 0, 1.06, 0.91), (1.04, 14, 0.99, 1.03), (1.16, 0, 1.02, 0.98),
        (1.26, 0, 1.00, 1.00)]


def at(t):
    for (t0, *a), (t1, *b) in zip(KEYS, KEYS[1:]):
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0)
            u = 0.5 - 0.5 * math.cos(math.pi * u)
            return [x + (y - x) * u for x, y in zip(a, b)]
    return KEYS[-1][1:]


def main():
    src, bg_path, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    target_w = float(sys.argv[4]) if len(sys.argv) > 4 else 1250
    frames = sorted(src.glob("*.png"))
    bg = Image.open(bg_path).convert("RGBA").resize((1920, 1080))
    first = Image.open(frames[0]).convert("RGBA")
    x0, y0, x1, y1 = first.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    k = target_w / (x1 - x0)
    tmp = Path(tempfile.mkdtemp())
    for n, f in enumerate(frames):
        t = n / FPS
        lift, sx, sy = at(t)
        im = Image.open(f).convert("RGBA")
        a = im.getchannel("A")
        # split the model from its contact shadow: the shadow is the faint, low-alpha part under the car
        body = im.copy(); body.putalpha(a.point(lambda v: 255 if v > 90 else int(v * 0.0)))
        shadow = im.copy(); shadow.putalpha(a.point(lambda v: v if v <= 90 else 0))
        bw, bh = int(im.width * k * sx), int(im.height * k * sy)
        body = body.resize((bw, bh), Image.LANCZOS)
        base_y = 540 + (y1 - im.height / 2) * k            # the car's ground line on screen
        cx = 960 - ((x0 + x1) / 2 - im.width / 2) * k
        frame = bg.copy()
        # shadow: stays on the ground, shrinks and fades while the car is up
        s = 1 - 0.25 * min(1.0, lift / 95)
        sh = shadow.resize((int(im.width * k * s), int(im.height * k * s)), Image.LANCZOS)
        sh.putalpha(sh.getchannel("A").point(lambda v: int(v * (1 - 0.5 * min(1.0, lift / 95)))))
        frame.alpha_composite(sh, (int(cx - sh.width / 2), int(base_y - (y1 - im.height / 2) * k * s - sh.height / 2)))
        # body: squash/stretch about its bottom edge, lifted
        top = base_y - lift - (y1 * k * sy)
        frame.alpha_composite(body, (int(cx - bw / 2), int(top)))
        frame.convert("RGB").save(tmp / f"{n:05d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "%05d.png"), "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    print(len(frames), "frames ->", out)


if __name__ == "__main__":
    main()
