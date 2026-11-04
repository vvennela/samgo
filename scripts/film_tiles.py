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
      catcher = new THREE.Mesh(new THREE.PlaneGeometry(R * 6, R * 6), new THREE.ShadowMaterial({opacity: window.__shadowOpacity || 0.22}));
      catcher.rotation.x = -Math.PI / 2; catcher.position.set(c.x, bb.min.y - 0.05, c.z); catcher.receiveShadow = true; catcher.userData.catcher = true;
      scene.add(catcher);
      sun.position.set(c.x + R * 0.35, c.y + R * 2.2, c.z + R * 0.5); sun.target.position.copy(c); scene.add(sun.target);
      sun.castShadow = true; sun.shadow.mapSize.set(4096, 4096); sun.shadow.radius = 6; sun.shadow.bias = -0.0005;
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


MESH = """(d) => {
  const dec = (b, T) => { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); };
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(dec(d.pos, Float32Array), 3));
  g.setAttribute('color', new THREE.BufferAttribute(dec(d.col, Float32Array), 3));
  g.setIndex(new THREE.BufferAttribute(dec(d.idx, Uint32Array), 1));
  g.computeVertexNormals();
  const mat = new THREE.MeshLambertMaterial({vertexColors: true, transparent: true, side: THREE.DoubleSide});
  const mesh = new THREE.Mesh(g, mat); mesh.userData.catcher = true;  // not a floor: never hidden
  scene.add(mesh); window.__mesh = mesh;
}"""


STYLE = """() => {
  // Film look: physically based plastic with a little sheen, a soft key and fill, faint outlines.
  renderer.physicallyCorrectLights = false;
  scene.traverse(o => { if (o.isLight && !o.userData.film) o.visible = false; });
  const add = l => { l.userData.film = true; scene.add(l); return l; };
  add(new THREE.HemisphereLight(0xffffff, 0xc9ced6, 0.62));
  const fill = add(new THREE.DirectionalLight(0xe4ecff, 0.28)); fill.position.set(-500, 300, -400);
  sun.visible = true; sun.userData.film = true; sun.intensity = 0.8; sun.color.set(0xfff6ec);
  window.__restyle = () => scene.traverse(o => {
    if (o.userData.styled || o.userData.catcher) return;
    if ((o.isMesh || o.isInstancedMesh) && o.material && o.material.isMeshLambertMaterial) {
      const m = o.material, s = new THREE.MeshStandardMaterial({color: m.color, roughness: 0.38, metalness: 0.0,
        transparent: m.transparent, opacity: m.opacity});
      if (m.transparent) { s.roughness = 0.08; s.opacity = Math.min(m.opacity, 0.5); }
      o.material = s; o.userData.styled = true;
    } else if (o.isLineSegments && o.material) {
      o.material = o.material.clone(); o.material.transparent = true; o.material.opacity = 0.16; o.userData.styled = true;
    }
  });
}"""


AO_SCRIPTS = ["shaders/CopyShader.js", "postprocessing/EffectComposer.js", "postprocessing/RenderPass.js",
              "postprocessing/ShaderPass.js", "math/SimplexNoise.js", "shaders/SSAOShader.js", "postprocessing/SSAOPass.js"]
AO_SETUP = """([w, h, radius]) => {
  const c = new THREE.EffectComposer(renderer);
  const p = new THREE.SSAOPass(scene, camera, w, h);
  p.kernelRadius = radius; p.minDistance = 0.00004; p.maxDistance = 0.012;
  p.output = THREE.SSAOPass.OUTPUT.Blur;
  c.addPass(p); window.__ao = c; window.__aoPass = p;
}"""


def ao_merge(color_png: Path, ao_png: Path, strength: float, ground: float) -> None:
    """Multiply the occlusion into the colour; where there is no model, occlusion becomes a soft dark film (contact
    darkening on the ground)."""
    import numpy as np
    from PIL import Image
    c = np.asarray(Image.open(color_png).convert("RGBA")).astype(np.float32)
    ao = np.asarray(Image.open(ao_png).convert("L")).astype(np.float32) / 255
    ao = 1 - strength * (1 - ao)
    a = c[..., 3:4] / 255
    rgb = c[..., :3] * ao[..., None]
    extra = np.clip((1 - ao) * ground, 0, 1)[..., None] * (1 - a)        # occlusion on the ground
    out_a = a + extra
    out_rgb = np.where(out_a > 1e-4, (rgb * a) / np.maximum(out_a, 1e-4), 0)
    out = np.concatenate([out_rgb, out_a * 255], axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(color_png)


def keyed(spec: str, t: float, smooth: bool = False) -> float:
    """Value at t from "t0:v0,t1:v1,..." (held flat outside the range): linear, or eased in-out with smooth."""
    keys = [tuple(float(x) for x in kv.split(":")) for kv in spec.split(",")]
    if t <= keys[0][0]:
        return keys[0][1]
    for (a, va), (b, vb) in zip(keys, keys[1:]):
        if a <= t <= b:
            u = (t - a) / max(1e-9, b - a)
            if smooth:
                u = u * u * (3 - 2 * u)
            return va + (vb - va) * u
    return keys[-1][1]


def load(path: Path):
    parts = json.loads(path.read_text())
    parts = parts["parts"] if isinstance(parts, dict) else parts
    manual = path.with_name("manual.json")
    return parts, (json.loads(manual.read_text()) if manual.exists() else None)


def main():
    out, model, secs = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    opts = dict(a.split("=", 1) for a in sys.argv[4:])
    w, h = (int(v) for v in opts.pop("size", "1920x1080").split("x"))
    ss = float(opts.pop("ss", "2"))         # supersampling: the same layout at ss x the pixels
    gpu = opts.pop("gpu", "0") == "1"
    building = opts.pop("build", "0") == "1"
    speed = float(opts.pop("build_speed", "1"))
    start = float(opts.pop("start", "0"))
    build_start = float(opts.pop("build_start", "0"))
    mesh = opts.pop("mesh", None)
    mesh_alpha = opts.pop("mesh_alpha", "0:1")
    style = opts.pop("style", "1") == "1"
    shadow_opacity = float(opts.pop("shadow", "0.16"))
    plan = json.loads(Path(opts.pop("plan")).read_text()) if "plan" in opts else None
    az_keys = opts.pop("az_keys", None)
    lift = float(opts.pop("lift", "90"))
    ao = opts.pop("ao", "1") == "1"
    ao_radius = float(opts.pop("ao_radius", "10"))  # mm
    ao_strength = float(opts.pop("ao_strength", "0.85"))
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
        pg = br.new_page(viewport={"width": int(w * ss), "height": int(h * ss)})
        pg.goto(html.as_uri())
        pg.wait_for_function("typeof window.renderAt === 'function'", timeout=60000)
        pg.add_style_tag(content=HIDE)
        pg.wait_for_timeout(300)
        pg.evaluate("() => { document.documentElement.style.overflow = 'hidden'; renderer.setPixelRatio(1); resize(); }")
        pg.wait_for_timeout(200)
        pg.evaluate(CLEAR)
        if style:
            pg.evaluate(STYLE)
        pg.evaluate("(v) => { window.__shadowOpacity = v; }", shadow_opacity)
        if mesh:
            pg.evaluate(MESH, json.loads(Path(mesh).read_text()))
        if plan:   # our own step order and arrival directions; Spark's markers as red rings
            pg.evaluate("""(p) => {
              MODELS[0].order = p.steps; MODELS[0].dirs = p.dirs; show(0, true); MODELS[0]._bb = null; window.EDGES = false;
              window.__rings = p.rings.map(r => {
                const m = new THREE.Mesh(new THREE.TorusGeometry(r.r, 1.1, 12, 64),
                  new THREE.MeshBasicMaterial({color: 0xd01012, transparent: true, opacity: 0}));
                m.position.set(r.x, r.up, r.z); m.userData = {t0: r.t0, t1: r.t1, catcher: true, styled: true};
                scene.add(m); return m;
              });
            }""", plan)
        if ao:
            for f in AO_SCRIPTS:
                pg.add_script_tag(url=f"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/{f}")
            pg.evaluate(AO_SETUP, [int(w * ss), int(h * ss), ao_radius])
        t0 = time.time()
        n = max(1, round(secs * FPS))
        for k in range(n):
            t = start + k / FPS
            frame = dict(o)
            if building:
                if t < build_start:
                    frame["step"] = -1                       # nothing built yet
                else:
                    frame["build"] = (t - build_start) * speed
            if az_keys:
                frame["az"] = keyed(az_keys, t, smooth=True)
            if plan:
                frame["step"], frame["lift"] = 0, 0
                for ev in plan["events"]:
                    if t >= ev["t"] - ev["dur"]:
                        u = min(1.0, (t - (ev["t"] - ev["dur"])) / ev["dur"])
                        frame["step"], frame["lift"] = ev["step"], lift * (1 - (1 - (1 - u) ** 3))
                pg.evaluate("""(t) => window.__rings.forEach(m => {
                  const a = Math.min(1, Math.max(0, (t - m.userData.t0) / 0.35)) * Math.min(1, Math.max(0, (m.userData.t1 - t) / 0.4));
                  m.material.opacity = a; m.visible = a > 0.01;
                  const s = 1 + 0.35 * Math.max(0, 1 - (t - m.userData.t0) / 0.35); m.scale.set(s, s, s);
                })""", t)
            if mesh:
                a = keyed(mesh_alpha, t)
                pg.evaluate("(a) => { const m = window.__mesh; m.visible = a > 0.01; m.material.opacity = a; "
                            "m.material.depthWrite = a > 0.97; m.castShadow = a > 0.5; }", a)
            pg.evaluate("([t, o]) => { const r = window.renderAt(0, t, o); if (window.__ao) { const R = MODELS[0]._bb.getSize(new THREE.Vector3()).length(); camera.near = R * 0.05; camera.far = R * 8; camera.updateProjectionMatrix(); } window.__hide(); window.__shadows(); if (window.__restyle) window.__restyle(); "
                        "renderer.render(scene, camera); return r; }", [t, frame])
            pg.locator("#stage canvas").screenshot(path=str(out / f"{k:05d}.png"), omit_background=True)
            if ao:
                pg.evaluate("() => { const cc = renderer.getClearColor(new THREE.Color()); renderer.setClearColor(0xffffff, 1); "
                            "window.__ao.render(); renderer.setClearColor(cc, 0); }")
                tmp = out / "_ao.png"
                pg.locator("#stage canvas").screenshot(path=str(tmp))
                if k == 0:
                    import shutil; shutil.copy(out / f"{k:05d}.png", out / "_color0.png")
                ao_merge(out / f"{k:05d}.png", tmp, ao_strength, 0.0)
                tmp.rename(out / f"_ao{k:05d}.png") if k == 0 else tmp.unlink()
        print(f"{n} frames in {time.time() - t0:.1f}s -> {out}", flush=True)
        br.close()


if __name__ == "__main__":
    main()
