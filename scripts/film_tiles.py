"""3D shots for the launch film: frames of a build from the 3D viewer (samgo.viewer3d) with no background and no
floor, saved as transparent PNGs so the models sit directly on the film's canvas (scripts/film.py).

    PYTHONPATH=. python scripts/film_tiles.py out_dir model.json seconds [key=value ...]

Options are passed to the viewer's renderAt per frame: build=1 plays the build animation from t=0 (build_speed
scales it), orbit=<seconds per turn>, dist, elev, az, move=0 holds every joint still. size=WxH (default 1920x1080).
"""
import json
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from samgo.viewer3d import facts_for, page

FPS = 30
HIDE = ("header, .bar, .facts { display: none !important; } html, body { background: transparent !important; "
        "padding: 0 !important; gap: 0 !important; } #stage { border-radius: 0 !important; height: 100vh !important; "
        "background: transparent !important; }")
CLEAR = """() => {
  // transparent clear, no floor plane, softer light from above
  renderer.setClearColor(0x000000, 0); scene.background = null; window.bg = () => {};
  const hide = () => scene.traverse(o => { if (o.isMesh && o.geometry && o.geometry.type === 'PlaneGeometry') o.visible = false; });
  hide(); window.__hide = hide;
}"""


def load(path: Path):
    parts = json.loads(path.read_text())
    parts = parts["parts"] if isinstance(parts, dict) else parts
    manual = path.with_name("manual.json")
    return parts, (json.loads(manual.read_text()) if manual.exists() else None)


def main():
    out, model, secs = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    opts = dict(a.split("=", 1) for a in sys.argv[4:])
    w, h = (int(v) for v in opts.pop("size", "1920x1080").split("x"))
    gpu = opts.pop("gpu", "0") == "1"
    building = opts.pop("build", "0") == "1"
    speed = float(opts.pop("build_speed", "1"))
    start = float(opts.pop("start", "0"))
    o = {k: (False if v == "false" else float(v)) for k, v in opts.items()}
    if "move" in o:
        o["move"] = bool(o["move"])
    out.mkdir(parents=True, exist_ok=True)
    parts, manual = load(model)
    html = Path(tempfile.mkdtemp()) / "tile.html"
    html.write_text(page([(model.stem, parts, facts_for(parts), manual)]))
    args = [] if gpu else ["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"]
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=args)
        pg = br.new_page(viewport={"width": w, "height": h})
        pg.goto(html.as_uri())
        pg.wait_for_function("typeof window.renderAt === 'function'", timeout=60000)
        pg.add_style_tag(content=HIDE)
        pg.wait_for_timeout(300)
        pg.evaluate(CLEAR)
        t0 = time.time()
        n = max(1, round(secs * FPS))
        for k in range(n):
            t = start + k / FPS
            frame = dict(o)
            if building:
                frame["build"] = t * speed
            pg.evaluate("([t, o]) => { const r = window.renderAt(0, t, o); window.__hide(); renderer.render(scene, camera); return r; }",
                        [t, frame])
            pg.locator("#stage canvas").screenshot(path=str(out / f"{k:05d}.png"), omit_background=True)
        print(f"{n} frames in {time.time() - t0:.1f}s -> {out}", flush=True)
        br.close()


if __name__ == "__main__":
    main()
