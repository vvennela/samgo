"""Join clips with the stud dissolve (scripts/film_dissolve.py) at every cut: clip A's last frame breaks into studs,
the studs flip, and clip B's first frame resolves out of them.

    python scripts/film_join.py out.mp4 a.mp4 b.mp4 ...
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from film_dissolve import frames  # noqa: E402

N = 20          # dissolve frames (0.73 s)


def grab(clip: Path, last: bool, out: Path) -> Image.Image:
    args = ["ffmpeg", "-v", "error", "-y"] + (["-sseof", "-0.05"] if last else []) + ["-i", str(clip), "-frames:v", "1",
                                                                                      str(out)]
    subprocess.run(args, check=True)
    return Image.open(out).convert("RGB").resize((1920, 1080))


def main():
    out, clips = Path(sys.argv[1]), [Path(c) for c in sys.argv[2:]]
    tmp = Path(tempfile.mkdtemp())
    parts = []
    for i, c in enumerate(clips):
        parts.append(c)
        if i + 1 < len(clips):
            a = grab(c, True, tmp / f"a{i}.png")
            b = grab(clips[i + 1], False, tmp / f"b{i}.png")
            d = tmp / f"d{i}"
            d.mkdir()
            for k, f in enumerate(frames(a, b, N, 30)):
                Image.fromarray(f.astype(np.uint8)).save(d / f"{k:05d}.png")
            clip = tmp / f"d{i}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(d / "%05d.png"), "-c:v",
                            "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(clip)], check=True)
            parts.append(clip)
    lst = tmp / "list.txt"
    norm = []
    for i, p in enumerate(parts):                      # same size, rate and format for the concat
        q = tmp / f"n{i:03d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(p), "-vf", "scale=1920:1080,fps=30", "-an", "-c:v",
                        "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(q)], check=True)
        norm.append(q)
    lst.write_text("".join(f"file '{q}'\n" for q in norm))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)],
                   check=True)
    print("joined", len(clips), "clips with", len(clips) - 1, "stud dissolves ->", out)


if __name__ == "__main__":
    main()
