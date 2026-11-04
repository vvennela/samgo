"""Screen-record the real SAMgo 3D viewer being used: a visible cursor clicks each model's tab, drags to turn it, and
presses "Preview motion" where asked. Real product footage for the film.

    PYTHONPATH=. python scripts/viewer_tour.py out.mp4 "Name=build.json[:motion]" ...
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from samgo.viewer3d import facts_for, page

CURSOR = """() => {
  const c = document.createElement('div'); c.id = '__cur';
  c.style.cssText = 'position:fixed;left:0;top:0;width:26px;height:26px;z-index:99999;pointer-events:none;transition:transform .05s';
  c.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24"><path d="M3 2l7 19 2.5-7.5L20 11z" fill="#111" stroke="#fff" stroke-width="1.4"/></svg>';
  document.body.appendChild(c);
  document.addEventListener('mousemove', e => { c.style.transform = `translate(${e.clientX - 3}px, ${e.clientY - 2}px)`; }, true);
  const r = document.createElement('div'); r.id = '__ring'; r.style.cssText = 'position:fixed;width:34px;height:34px;border-radius:50%;border:3px solid rgba(208,16,18,.75);z-index:99998;pointer-events:none;opacity:0;transition:opacity .35s, transform .35s';
  document.body.appendChild(r);
  document.addEventListener('mousedown', e => { r.style.left = (e.clientX - 17) + 'px'; r.style.top = (e.clientY - 17) + 'px'; r.style.opacity = 1; r.style.transform = 'scale(.6)';
    setTimeout(() => { r.style.opacity = 0; r.style.transform = 'scale(1.4)'; }, 60); }, true);
}"""


def glide(pg, x, y, steps=28):
    pg.mouse.move(x, y, steps=steps)


def main():
    out = Path(sys.argv[1]).resolve()
    specs = []
    for a in sys.argv[2:]:
        name, rest = a.split("=", 1)
        path, *flags = rest.split(":")
        az = next((float(f[3:]) for f in flags if f.startswith("az=")), 0.6)
        dist = next((float(f[5:]) for f in flags if f.startswith("dist=")), 1.05)
        specs.append((name, Path(path), "motion" in flags, (az, dist)))
    models = []
    for name, path, _, _ in specs:
        parts = json.loads(path.read_text())
        parts = parts["parts"] if isinstance(parts, dict) else parts
        manual = path.with_name("manual.json")
        models.append((name, parts, facts_for(parts), json.loads(manual.read_text()) if manual.exists() else None))
    html = Path(tempfile.mkdtemp()) / "viewer.html"
    html.write_text(page(models))
    vids = Path(tempfile.mkdtemp())
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"])
        ctx = br.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(vids),
                             record_video_size={"width": 1920, "height": 1080}, color_scheme="light")
        pg = ctx.new_page()
        pg.goto(html.as_uri())
        pg.wait_for_function("typeof window.renderAt === 'function'", timeout=60000)
        pg.add_style_tag(content="body{font-size:20px} .bar button{font-size:1.05rem;padding:10px 16px} h1{font-size:1.6rem}")
        pg.evaluate(CURSOR)
        pg.mouse.move(960, 600)
        pg.wait_for_timeout(800)
        for i, (name, _, motion, az) in enumerate(specs):
            tab = pg.locator("#models button").nth(i)
            box = tab.bounding_box()
            glide(pg, box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            pg.wait_for_timeout(250)
            tab.click()
            pg.wait_for_timeout(500)
            # the front three-quarter, close in (the viewer's own camera and controls)
            pg.evaluate("""([az, k]) => { const bb = bounds(group), c = bb.getCenter(new THREE.Vector3()), s = bb.getSize(new THREE.Vector3());
              const d = Math.max(s.x, s.y, s.z) * k; camera.position.set(c.x + Math.cos(az) * d, c.y + 0.45 * d, c.z + Math.sin(az) * d);
              controls.target.copy(c); controls.update(); }""", list(az))
            pg.wait_for_timeout(900)
            # a small turn with the mouse, staying on the front
            glide(pg, 1000, 640, 18)
            pg.mouse.down()
            glide(pg, 965, 636, 40)
            pg.mouse.up()
            pg.wait_for_timeout(500)
            if motion and pg.locator("#sim").is_visible():
                b = pg.locator("#sim").bounding_box()
                glide(pg, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
                pg.wait_for_timeout(200)
                pg.locator("#sim").click()
                pg.wait_for_timeout(4200)
                pg.locator("#sim").click()           # motion off before the next model
            else:
                pg.wait_for_timeout(1800)
        pg.wait_for_timeout(600)
        ctx.close()
        br.close()
    webm = max(vids.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1.6", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    str(out)], check=True)
    print("recorded", out, flush=True)


if __name__ == "__main__":
    main()
