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
  const hide = () => scene.traverse(o => { if (o.isMesh && o.geometry && o.geometry.type === 'PlaneGeometry' && !o.userData.catcher) o.visible = false; });
  hide(); window.__hide = hide;
  // soft contact shadows on an invisible ground that only catches shadow (sized to the finished model)
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  let catcher = null;
  window.__shadows = () => {
    const m = MODELS[0];
    if (!catcher && m._bb) {
      const bb = m._bb, c = bb.getCenter(new THREE.Vector3()), s = bb.getSize(new THREE.Vector3());
      const R = Math.max(s.x, s.y, s.z);
      catcher = new THREE.Mesh(new THREE.PlaneGeometry(R * 6, R * 6), new THREE.ShadowMaterial({opacity: 0.22}));
      catcher.rotation.x = -Math.PI / 2; catcher.position.set(c.x, bb.min.y - 0.05, c.z); catcher.receiveShadow = true; catcher.userData.catcher = true;
      scene.add(catcher);
      sun.position.set(c.x + R * 0.35, c.y + R * 2.2, c.z + R * 0.5); sun.target.position.copy(c); scene.add(sun.target);
      sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.radius = 6; sun.shadow.bias = -0.0005;
      const sc = sun.shadow.camera; sc.left = sc.bottom = -R * 1.2; sc.right = sc.top = R * 1.2; sc.near = 1; sc.far = R * 6;
      sc.updateProjectionMatrix();
    }
    scene.traverse(o => {
      if ((o.isMesh || o.isInstancedMesh) && o !== catcher && !o.castShadow) {
        o.castShadow = true;
        (Array.isArray(o.material) ? o.material : [o.material]).forEach(mt => { mt.needsUpdate = true; });
      }
    });
    if (catcher && !catcher.material.__ok) { catcher.material.needsUpdate = true; catcher.material.__ok = true; }
  };
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
            pg.evaluate("([t, o]) => { const r = window.renderAt(0, t, o); window.__hide(); window.__shadows(); "
                        "renderer.render(scene, camera); return r; }", [t, frame])
            pg.locator("#stage canvas").screenshot(path=str(out / f"{k:05d}.png"), omit_background=True)
        print(f"{n} frames in {time.time() - t0:.1f}s -> {out}", flush=True)
        br.close()


if __name__ == "__main__":
    main()
