"""The launch film's compositor: one continuous canvas, one camera, no cuts.

Every element is pinned on a large canvas (world pixels); a camera with eased keyframes glides and zooms between them.
Clips play from the moment given in the spec, 3D shots (transparent frames from scripts/film_tiles.py) sit directly on
the canvas, rectangular media get rounded corners and a soft shadow. The background is a soft colour with a faint
stud-dot grid that moves with the camera, so every move reads as travel across one surface.

    PYTHONPATH=. python scripts/film.py spec.py out.mp4 [--from S --to S --scale 0.5 --workers 8]

The spec is a Python file defining W, H, FPS, DURATION, BG, ELEMENTS (list of dicts) and CAMERA (list of keyframes);
see scripts/film_spec.py.
"""
import argparse
import math
import runpy
import subprocess
import tempfile
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SPEC = {}


# ---------------------------------------------------------------- easing and camera
def ease(u: float, kind: str = "inout") -> float:
    u = min(1.0, max(0.0, u))
    if kind == "linear":
        return u
    if kind == "out":
        return 1 - (1 - u) ** 3
    if kind == "in":
        return u ** 3
    return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2      # cubic in-out


def camera(t: float):
    """(cx, cy, zoom) at time t. Keyframes: dict(t, x, y, zoom, ease='inout', arc=0). Between two keyframes the
    position eases, zoom eases in log space, and arc > 0 pulls the zoom out mid-move (a flight over the canvas)."""
    keys = SPEC["CAMERA"]
    if t <= keys[0]["t"]:
        k = keys[0]
        return k["x"], k["y"], k["zoom"]
    for a, b in zip(keys, keys[1:]):
        if a["t"] <= t <= b["t"]:
            u = ease((t - a["t"]) / max(1e-6, b["t"] - a["t"]), b.get("ease", "inout"))
            x = a["x"] + (b["x"] - a["x"]) * u
            y = a["y"] + (b["y"] - a["y"]) * u
            lz = math.log(a["zoom"]) + (math.log(b["zoom"]) - math.log(a["zoom"])) * u
            lz -= b.get("arc", 0) * math.sin(math.pi * u)
            return x, y, math.exp(lz)
    k = keys[-1]
    return k["x"], k["y"], k["zoom"]


# ---------------------------------------------------------------- sources
@lru_cache(maxsize=64)
def _frames(path: str) -> list[str]:
    p = Path(path)
    return sorted(str(f) for f in p.glob("*.png")) if p.is_dir() else [path]


@lru_cache(maxsize=48)
def _load(path: str, radius: float) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    if radius:
        r = int(radius * im.width)
        mask = Image.new("L", im.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, im.width - 1, im.height - 1], r, fill=255)
        a = im.getchannel("A")
        im.putalpha(Image.composite(a, Image.new("L", im.size, 0), mask))
    return im


@lru_cache(maxsize=32)
def _shadow(w: int, h: int, radius: float, blur: float) -> Image.Image:
    """A soft drop shadow for a w x h rounded rectangle, padded by 3 blur radii."""
    pad = int(3 * blur)
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    ImageDraw.Draw(im).rounded_rectangle([pad, pad, pad + w, pad + h], int(radius * w), fill=70)
    im = im.filter(ImageFilter.GaussianBlur(blur))
    out = Image.new("RGBA", im.size, (20, 30, 45, 0))
    out.putalpha(im)
    return out


def frame_of(e: dict, t: float):
    """The source image of element e at film time t, or None before it appears / after it is gone."""
    files = _frames(e["src"])
    if not files or not Path(files[0]).exists():
        return None                 # not rendered yet
    if len(files) == 1:
        return files[0]
    if "schedule" in e:   # stop motion: file i shows from start + schedule[i]
        import bisect
        k = bisect.bisect_right(e["schedule"], t - e.get("start", 0)) - 1
        return files[max(0, min(len(files) - 1, k))]
    k = int(round((t - e.get("start", 0)) * e.get("fps", 30) * e.get("rate", 1)))
    if e.get("loop"):
        k %= len(files)
    return files[max(0, min(len(files) - 1, k))]


def pop(e: dict, t: float):
    """(alpha, scale, dy) of a pop entrance at e["pop"]: fades in fast, settles from 108% with a small overshoot,
    rises 30 px into place."""
    if "pop" not in e:
        return 1.0, 1.0, 0.0
    u = (t - e["pop"]) / e.get("pop_dur", 0.45)
    if u <= 0:
        return 0.0, 1.0, 0.0
    u = min(1.0, u)
    c1, c3 = 1.4, 2.4
    back = 1 + c3 * (u - 1) ** 3 + c1 * (u - 1) ** 2       # ease-out-back, 0 -> 1 with a small overshoot
    return min(1.0, u * 3), 1.08 - 0.08 * back, 30 * (1 - ease(u, "out"))


def opacity(e: dict, t: float) -> float:
    a = pop(e, t)[0]
    if "show" in e:
        t_in, t_out = e["show"]
        f = e.get("fade", 0.4)
        a = min(a, ease((t - t_in) / f, "out") if f else float(t >= t_in),
                ease((t_out - t) / f, "out") if f else float(t <= t_out))
    return max(0.0, min(1.0, a))


# ---------------------------------------------------------------- drawing
@lru_cache(maxsize=2)
def _bg_image(path: str, w: int, h: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)


def background(cx: float, cy: float, z: float) -> Image.Image:
    W, H = SPEC["W"], SPEC["H"]
    if SPEC.get("BG_IMAGE"):
        return _bg_image(SPEC["BG_IMAGE"], W, H).copy()
    im = Image.new("RGB", (W, H), SPEC["BG"])
    grid = SPEC.get("GRID", 64)
    step = grid * z
    if step >= 9:     # dots fade out when the camera is far away
        d = ImageDraw.Draw(im)
        r = max(1.0, 2.2 * z ** 0.5)
        c = SPEC.get("DOT", (214, 220, 228))
        x0 = (W / 2 - cx * z) % step
        y0 = (H / 2 - cy * z) % step
        y = y0 - step
        while y < H + step:
            x = x0 - step
            while x < W + step:
                d.ellipse([x - r, y - r, x + r, y + r], fill=c)
                x += step
            y += step
    return im


def render(k: int) -> Image.Image:
    W, H = SPEC["W"], SPEC["H"]
    t = SPEC["T0"] + k / SPEC["FPS"]
    cx, cy, z = camera(t)
    im = background(cx, cy, z).convert("RGBA")
    for e in SPEC["ELEMENTS"]:
        a = opacity(e, t)
        if a <= 0.004:
            continue
        path = frame_of(e, t)
        if path is None:
            continue
        src = _load(path, e.get("radius", 0))
        _, scale, dy = pop(e, t)
        ww = e["w"] * scale
        wh = ww * src.height / src.width
        if e.get("screen"):   # fixed on screen (x, y, w in 1920x1080 frame pixels), not on the canvas
            k = W / 1920
            sx, sy, sw, sh = e["x"] * k, e["y"] * k, ww * k, wh * k
        else:
            sx = (e["x"] - cx) * z + W / 2
            sy = (e["y"] + dy - cy) * z + H / 2
            sw, sh = ww * z, wh * z
        if sx + sw / 2 < 0 or sx - sw / 2 > W or sy + sh / 2 < 0 or sy - sh / 2 > H or sw < 2:
            continue
        # crop to what is on screen before scaling (a zoomed-in tile only scales its visible part)
        left, top = sx - sw / 2, sy - sh / 2
        fx0 = max(0.0, -left / sw); fy0 = max(0.0, -top / sh)
        fx1 = min(1.0, (W - left) / sw); fy1 = min(1.0, (H - top) / sh)
        box = (fx0 * src.width, fy0 * src.height, fx1 * src.width, fy1 * src.height)
        out_w = max(1, round((fx1 - fx0) * sw)); out_h = max(1, round((fy1 - fy0) * sh))
        if e.get("shadow") and sw > 40:
            sh_img = _shadow(max(1, round(sw)), max(1, round(sh)), e.get("radius", 0), max(4.0, 0.025 * sw))
            pad = (sh_img.width - round(sw)) // 2
            if a < 1:
                sh_img = sh_img.copy(); sh_img.putalpha(sh_img.getchannel("A").point(lambda v: int(v * a)))
            im.alpha_composite(sh_img, (round(left) - pad, round(top) - pad + round(0.012 * sw)))
        piece = src.resize((out_w, out_h), Image.LANCZOS if out_w < src.width / 2 else Image.BILINEAR, box=box)
        if a < 1:
            piece.putalpha(piece.getchannel("A").point(lambda v: int(v * a)))
        im.alpha_composite(piece, (round(left + fx0 * sw), round(top + fy0 * sh)))
    return im.convert("RGB")


def work(args):
    spec_path, ks, outdir, overrides = args
    SPEC.update(runpy.run_path(spec_path)); SPEC.update(overrides)
    for k in ks:
        render(k).save(Path(outdir) / f"{k:06d}.png", compress_level=1)
    return len(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("out")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="render smaller for previews")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--audio", default=None)
    a = ap.parse_args()
    SPEC.update(runpy.run_path(a.spec))
    t1 = a.t1 if a.t1 is not None else SPEC["DURATION"]
    W, H = int(SPEC["W"] * a.scale), int(SPEC["H"] * a.scale)
    overrides = {"W": W, "H": H, "T0": a.t0}
    if a.scale != 1:   # previews: the same film, every world size and camera zoom scaled down
        overrides["CAMERA"] = [dict(k, zoom=k["zoom"] * a.scale) for k in SPEC["CAMERA"]]
        overrides["GRID"] = SPEC.get("GRID", 64)
    n = int(round((t1 - a.t0) * SPEC["FPS"]))
    outdir = Path(tempfile.mkdtemp(prefix="film_"))
    chunks = [list(range(i, n, a.workers)) for i in range(a.workers)]
    with Pool(a.workers) as pool:
        done = sum(pool.map(work, [(a.spec, c, str(outdir), overrides) for c in chunks]))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(SPEC["FPS"]), "-i", str(outdir / "%06d.png")]
    if a.audio:
        cmd += ["-ss", str(a.t0), "-i", a.audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium", a.out]
    subprocess.run(cmd, check=True)
    print(f"{done} frames -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
