"""Key a flat background colour out of video frames (the 10x resolution clips): transparent PNGs whose edge pixels
have the background un-mixed, so they sit cleanly on the film's canvas.

    python scripts/film_key.py in_dir out_dir R,G,B
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

BG = None


def key(args):
    src, dst, bg = args
    a = np.asarray(Image.open(src).convert("RGB")).astype(float)
    d = np.sqrt(((a - bg) ** 2).sum(-1))
    alpha = np.clip((d - 5) / 9, 0, 1)[..., None]
    rgb = np.where(alpha > 0.02, (a - bg * (1 - alpha)) / np.maximum(alpha, 0.02), 0)
    Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha[..., 0] * 255]).astype(np.uint8), "RGBA").save(dst)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    bg = np.array([float(v) for v in sys.argv[3].split(",")])
    dst.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(dst / f.name), bg) for f in sorted(src.glob("*.png"))]
    with ProcessPoolExecutor(10) as ex:
        list(ex.map(key, jobs, chunksize=8))
    print(len(jobs), "frames ->", dst)


if __name__ == "__main__":
    main()
