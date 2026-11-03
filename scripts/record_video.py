"""Record video clips of builds from the 3D viewer (headless Chromium): every joint moving at once and the camera
orbiting, or the build animation; frames rendered at exact times, then encoded with ffmpeg.

    PYTHONPATH=. python scripts/record_video.py out_dir "Name=path/build.json:seconds[:build][:camera distance]" ...
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from samgo.viewer3d import facts_for, page

FPS, W, H = 30, 1920, 1080
HIDE = ("header, .bar, .facts { display: none !important; } body { padding: 0 !important; gap: 0 !important; } "
        "#stage { border-radius: 0 !important; height: 100vh !important; }")


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    specs = []
    for arg in sys.argv[2:]:
        name, rest = arg.split("=", 1)
        bits = rest.split(":")
        building = "build" in bits[2:]
        dist = next((float(b) for b in bits[2:] if b != "build"), None)
        specs.append((name, bits[0], float(bits[1]), building, dist))
    models = []
    for name, path, _, _, _ in specs:
        parts = json.loads(Path(path).read_text())
        parts = parts["parts"] if isinstance(parts, dict) else parts
        manual = Path(path).with_name("manual.json")
        models.append((name, parts, facts_for(parts), json.loads(manual.read_text()) if manual.exists() else None))
    html = Path(tempfile.mkdtemp()) / "video.html"
    html.write_text(page(models))
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"])
        pg = br.new_page(viewport={"width": W, "height": H}, color_scheme="dark")
        pg.goto(html.as_uri())
        pg.wait_for_function("typeof window.renderAt === 'function'", timeout=60000)
        pg.add_style_tag(content=HIDE)
        pg.wait_for_timeout(500)
        for i, (name, _, secs, building, dist) in enumerate(specs):
            frames = Path(tempfile.mkdtemp())
            n = int(secs * FPS)
            for k in range(n):
                t = k / FPS
                opts = {"build": t, "move": False, "orbit": 40, "dist": dist or 1.25} if building else {"dist": dist or 1.15}
                pg.evaluate("([i, t, o]) => window.renderAt(i, t, o)", [i, t, opts])
                pg.locator("#stage canvas").screenshot(path=str(frames / f"{k:05d}.png"))
            clip = out / f"{i + 1:02d}_{name.replace(' ', '_').replace('(', '').replace(')', '')}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%05d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(clip)], check=True)
            print(clip, n, "frames", flush=True)
        br.close()


if __name__ == "__main__":
    main()
