"""Video of a large model whole and coming apart into horizontal layers (headless Chromium + three.js, ffmpeg).

Each part goes to the layer its bottom is in (no part is cut). Only parts that can be seen are drawn: a part with any
face open to air, counting a layer's own cut faces as open (so the inside shows when the layers separate). Every
piece is drawn 0.2 mm smaller on each side and slightly varied in brightness, so the seams between parts show.

    PYTHONPATH=. python scripts/record_layers.py build.json out_dir [layers]
writes out_dir/whole.mp4 (8 s orbit) and out_dir/layers.mp4 (whole -> layers apart -> hold, 12 s)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from samgo.parts import COLORS, GLASS_COLORS

STUD, PLATE, FPS, W, H = 8.0, 3.2, 30, 1920, 1080
HEIGHT = {"plate": 1, "brick": 3}
GLASS_RGB = (70, 110, 150)


def visible(parts: list[dict]) -> list[int]:
    """Parts with any cell next to air, within these parts only."""
    lo = np.min([[q["x"], q["y"], q["z"]] for q in parts], axis=0)
    hi = np.max([[q["x"] + q["w"], q["y"] + q["l"], q["z"] + HEIGHT[q["type"]]] for q in parts], axis=0)
    occ = np.full(tuple(hi - lo + 2), -1, np.int64)
    clear = np.zeros(occ.shape, bool)  # glass: seen through, so what is behind it shows
    for i, q in enumerate(parts):
        x, y, z = q["x"] - lo[0] + 1, q["y"] - lo[1] + 1, q["z"] - lo[2] + 1
        occ[x:x + q["w"], y:y + q["l"], z:z + HEIGHT[q["type"]]] = i
        if q.get("color") in GLASS_COLORS:
            clear[x:x + q["w"], y:y + q["l"], z:z + HEIGHT[q["type"]]] = True
    filled = occ >= 0
    opaque = filled & ~clear
    open_ = np.zeros_like(filled)
    for ax in range(3):
        for d in (1, -1):
            open_ |= ~np.roll(opaque, d, axis=ax)
    return sorted(set(np.unique(occ[filled & open_]).tolist()))


def colour(name: str) -> tuple:
    if name in GLASS_COLORS:
        return GLASS_RGB
    return COLORS.get(name, COLORS["light_gray"])[1]


PAGE = """<!doctype html><html><body style="margin:0;background:#0e1116;overflow:hidden">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const DATA = __DATA__;
const r = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
r.setSize(__W__, __H__); r.setPixelRatio(1); document.body.appendChild(r.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0e1116);
scene.add(new THREE.HemisphereLight(0xffffff, 0x303040, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(-1, 2.2, 1.4); scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff, 0.25); fill.position.set(1.5, 0.6, -1); scene.add(fill);
const cam = new THREE.PerspectiveCamera(30, __W__ / __H__, 10, 1e6);
const box = new THREE.BoxGeometry(1, 1, 1); const m4 = new THREE.Matrix4(); const c = new THREE.Color();
const layers = [];
for (const L of DATA.layers) {
  const g = new THREE.Group();
  for (const [key, boxes] of Object.entries(L.groups)) {
    const glass = key === "glass";
    const mat = new THREE.MeshStandardMaterial({roughness: 0.55, metalness: 0.0, transparent: glass,
                                                opacity: glass ? 0.62 : 1, depthWrite: !glass});
    const mesh = new THREE.InstancedMesh(box, mat, boxes.length);
    boxes.forEach((b, i) => {  // [x, y, z, w, l, h, r, g, b] in mm, y up
      m4.makeScale(b[3], b[5], b[4]); m4.setPosition(b[0] + b[3] / 2, b[2] + b[5] / 2, b[1] + b[4] / 2);
      mesh.setMatrixAt(i, m4); c.setRGB(b[6] / 255, b[7] / 255, b[8] / 255); mesh.setColorAt(i, c);
    });
    if (glass) mesh.renderOrder = 1;
    g.add(mesh);
  }
  scene.add(g); layers.push(g);
}
const C = DATA.centre, R = DATA.radius;
window.renderAt = (t, o) => {
  layers.forEach((g, k) => { g.position.y = k * o.sep * DATA.gap; });
  const a = o.angle0 + t * o.spin, lift = DATA.gap * (layers.length - 1) * o.sep / 2;
  cam.position.set(C[0] + Math.cos(a) * R * o.dist, C[1] + lift + R * o.dist * o.height, C[2] + Math.sin(a) * R * o.dist);
  cam.lookAt(C[0], C[1] + lift, C[2]); r.render(scene, cam);
};
window.ready = true;
</script></body></html>"""


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    out.mkdir(parents=True, exist_ok=True)
    parts = [q for q in json.loads(src.read_text()) if q["type"] in HEIGHT]
    Z = max(q["z"] + HEIGHT[q["type"]] for q in parts)
    edges = [round(k * Z / n) for k in range(n + 1)]
    rng = np.random.default_rng(0)
    layers, shown = [], 0
    for k in range(n):
        s = [q for q in parts if edges[k] <= q["z"] < edges[k + 1]]
        groups: dict = {}
        for i in visible(s):
            q = s[i]
            rgb = np.array(colour(q["color"]), float)
            if q["color"] not in GLASS_COLORS:
                rgb = np.clip(rgb * rng.uniform(0.9, 1.08), 0, 255)
            e = 0.2
            groups.setdefault("glass" if q["color"] in GLASS_COLORS else "solid", []).append(
                [round(q["x"] * STUD + e, 1), round(q["y"] * STUD + e, 1), round(q["z"] * PLATE + e, 2),
                 round(q["w"] * STUD - 2 * e, 1), round(q["l"] * STUD - 2 * e, 1),
                 round(HEIGHT[q["type"]] * PLATE - 2 * e, 2), *[int(v) for v in rgb]])
        shown += sum(len(v) for v in groups.values())
        layers.append({"groups": groups})
        print(f"layer {k + 1}: {len(s)} parts, {sum(len(v) for v in groups.values())} visible", flush=True)
    xs = [q["x"] * STUD for q in parts] + [(q["x"] + q["w"]) * STUD for q in parts]
    ys = [q["y"] * STUD for q in parts] + [(q["y"] + q["l"]) * STUD for q in parts]
    centre = [(min(xs) + max(xs)) / 2, Z * PLATE / 2, (min(ys) + max(ys)) / 2]
    radius = max(max(xs) - min(xs), max(ys) - min(ys), Z * PLATE) / 2
    data = {"layers": layers, "centre": centre, "radius": radius, "gap": Z * PLATE / n * 1.5}
    html = Path(tempfile.mkdtemp()) / "layers.html"
    html.write_text(PAGE.replace("__DATA__", json.dumps(data)).replace("__W__", str(W)).replace("__H__", str(H)))
    print(f"{shown} visible parts drawn; page {html.stat().st_size / 1e6:.1f} MB", flush=True)

    def ease(u):
        u = min(max(u, 0.0), 1.0)
        return u * u * (3 - 2 * u)

    shots = {
        "whole": (8.0, lambda t: {"sep": 0.0, "angle0": 0.6, "spin": 0.35, "dist": 2.6, "height": 0.45}),
        "layers": (12.0, lambda t: {"sep": ease((t - 2.0) / 5.0), "angle0": 1.0, "spin": 0.12,
                                    "dist": 2.6 + 1.9 * ease((t - 2.0) / 5.0), "height": 0.22}),
    }
    with sync_playwright() as pw:
        import os

        gpu = os.environ.get("GPU", "metal")  # the Mac's GPU; "swiftshader" = software, much slower
        br = pw.chromium.launch(headless=True, args=["--use-gl=angle", f"--use-angle={gpu}", "--ignore-gpu-blocklist",
                                                     "--enable-gpu-rasterization"])
        pg = br.new_page(viewport={"width": W, "height": H})
        pg.goto(html.as_uri())
        pg.wait_for_function("window.ready === true", timeout=600000)
        import os
        import time

        if os.environ.get("PREVIEW"):  # a few stills per shot instead of the clips
            for name, (secs, opts) in shots.items():
                for t in (0.0, secs * 0.5, secs - 0.1):
                    t0 = time.time()
                    pg.evaluate("([t, o]) => window.renderAt(t, o)", [t, opts(t)])
                    pg.locator("canvas").screenshot(path=str(out / f"preview_{name}_{t:.1f}.png"))
                    print("still", name, t, f"{time.time() - t0:.1f}s", flush=True)
            br.close()
            return
        for name, (secs, opts) in shots.items():
            frames = Path(tempfile.mkdtemp())
            for k in range(int(secs * FPS)):
                t = k / FPS
                pg.evaluate("([t, o]) => window.renderAt(t, o)", [t, opts(t)])
                pg.locator("canvas").screenshot(path=str(frames / f"{k:05d}.png"))
                if k % 60 == 0:
                    print(name, "frame", k, flush=True)
            clip = out / f"{name}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%05d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(clip)], check=True)
            print(clip, flush=True)
        br.close()


if __name__ == "__main__":
    main()
