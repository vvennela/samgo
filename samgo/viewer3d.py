"""Interactive 3D page for SAMgo builds: rotate, zoom, compare. Used whenever a model is shared.

Parts are drawn in millimetres at true LEGO scale (stud 8 mm, plate 3.2 mm), with real depth (WebGL), so wheels
and the body occlude each other correctly (matplotlib's 3D renders do not). Axles and wheels sit where the
function verifier puts them: on stud edges at Technic hole height.

    uv run python -m samgo.viewer3d out.html "Glimmer=runs/agent/running_gear/camry_parts.json" \
        "Algorithm=runs/s6/camry/build.json"
"""

import json
import sys
from pathlib import Path

from .parts import COLORS
from .technic import DEFAULT_WHEEL, HOLE_HEIGHT, PLATE, STUD, WHEELS, wheel_studs

HEIGHT = {"plate": 1, "brick": 3, "technic_brick": 3}
EXTRA = {"trans_clear": (230, 240, 245), "trans_black": (40, 50, 60)}


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(rgb)


def scene(parts: list[dict]) -> dict:
    """Boxes, axles and wheels in mm, y up (LEGO z), grouped for the page."""
    boxes, axles, wheels = [], [], []
    for p in parts:
        t = p.get("type")
        if t in HEIGHT:
            rgb = COLORS.get(p.get("color", "light_gray"), (None, EXTRA.get(p.get("color"), (150, 150, 150))))[1]
            boxes.append([p["x"] * STUD, p["z"] * PLATE, p["y"] * STUD, p["w"] * STUD, HEIGHT[t] * PLATE,
                          p["l"] * STUD, _hex(rgb), 1 if t == "technic_brick" else 0])
        elif t in ("axle", "wheel"):
            size = p.get("size", DEFAULT_WHEEL)
            span = (p["length"] if t == "axle" else wheel_studs(size)) * STUD
            centre_h = p["z"] * PLATE + HOLE_HEIGHT
            rec = [p["x"] * STUD, centre_h, p["y"] * STUD, span, p.get("axis", "y")]
            if t == "wheel":
                rec.append(WHEELS[size][0] / 2)       # radius
                rec.append(WHEELS[size][1])           # tyre width, mm
            (axles if t == "axle" else wheels).append(rec)
    return {"boxes": boxes, "axles": axles, "wheels": wheels}


PAGE = """<title>SAMgo Build Viewer</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a full-height stage with a slim control bar; the model is the page. */
:root {
  --bg: #eef0f2; --panel: #ffffff; --fg: #1d2329; --muted: #5d6873; --line: #d5dae0; --accent: #d01012;
  --stage: #e3e7eb;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #15191d; --panel: #1d2328; --fg: #e6eaee; --muted: #9aa5b0; --line: #2f373f; --accent: #ff4a3d; --stage: #101316; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #15191d; --panel: #1d2328; --fg: #e6eaee; --muted: #9aa5b0; --line: #2f373f; --accent: #ff4a3d; --stage: #101316; color-scheme: dark }
html, body { height: 100%; }
body { background: var(--bg); color: var(--fg); font-family: var(--display); display: flex; flex-direction: column;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 12px; box-sizing: border-box; gap: 10px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px; }
h1 { font-size: 1.15rem; margin: 0; letter-spacing: .01em; text-wrap: balance; }
h1 b { color: var(--accent); }
.note { color: var(--muted); font-size: .85rem; margin: 0; min-width: 0; }
.bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.bar button { font: 500 .85rem var(--display); border: 1px solid var(--line); background: var(--panel); color: var(--fg);
  padding: 7px 12px; border-radius: 6px; cursor: pointer; }
.bar button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
.bar button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.bar label { display: flex; gap: 6px; align-items: center; font-size: .85rem; color: var(--muted); }
#stage { flex: 1; min-height: 320px; border-radius: 8px; overflow: hidden; background: var(--stage); position: relative; }
#stage canvas { display: block; width: 100%; height: 100%; }
.facts { font: .8rem var(--mono); color: var(--muted); display: flex; flex-wrap: wrap; gap: 4px 18px; font-variant-numeric: tabular-nums; }
.facts span b { color: var(--fg); font-weight: 500; }
</style>
<header><h1>SAMgo <b>Build Viewer</b></h1><p class="note">Drag to rotate, scroll to zoom, right-drag to pan. True LEGO scale in mm.</p></header>
<div class="bar" id="models"></div>
<div class="bar">
  <label><input type="checkbox" id="hl"> Highlight running gear (Technic bricks, axles, wheels)</label>
  <label><input type="checkbox" id="xray"> See-through body</label>
  <button type="button" id="side">Side view</button><button type="button" id="three">3/4 view</button><button type="button" id="under">Underside</button>
</div>
<div id="stage"></div>
<div class="facts" id="facts"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const MODELS = __MODELS__;
const stage = document.getElementById("stage");
const renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
stage.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 1, 20000);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 0.75); sun.position.set(300, 600, 400); scene.add(sun);
let group = null, current = 0;

function build(m) {
  const g = new THREE.Group();
  const hl = document.getElementById("hl").checked, xray = document.getElementById("xray").checked;
  const box = new THREE.BoxGeometry(1, 1, 1);
  const byColor = {};
  for (const b of m.scene.boxes) (byColor[(hl && b[7]) ? "#ffb000" : b[6]] ||= []).push(b);
  const edgeMat = new THREE.LineBasicMaterial({color: 0x000000, transparent: true, opacity: 0.18});
  for (const [col, list] of Object.entries(byColor)) {
    const technic = hl && col === "#ffb000";
    const mat = new THREE.MeshLambertMaterial({color: col, transparent: xray && !technic, opacity: xray && !technic ? 0.18 : 1});
    const inst = new THREE.InstancedMesh(box, mat, list.length);
    const mtx = new THREE.Matrix4();
    list.forEach((b, i) => { mtx.makeScale(b[3] - 0.2, b[4] - 0.1, b[5] - 0.2); mtx.setPosition(b[0] + b[3] / 2, b[1] + b[4] / 2, b[2] + b[5] / 2); inst.setMatrixAt(i, mtx); });
    g.add(inst);
  }
  const axleMat = new THREE.MeshLambertMaterial({color: hl ? 0xffb000 : 0x7a7f86});
  for (const a of m.scene.axles) {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(2.4, 2.4, a[3], 16), axleMat);
    orient(c, a); g.add(c);
  }
  const tyre = new THREE.MeshLambertMaterial({color: 0x1b1b1b}), rim = new THREE.MeshLambertMaterial({color: hl ? 0xffb000 : 0xc9ccd1});
  for (const w of m.scene.wheels) {
    const r = w[5], tw = w[6];
    const t = new THREE.Mesh(new THREE.CylinderGeometry(r, r, tw, 40), tyre); orient(t, w); g.add(t);
    const h = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.6, r * 0.6, tw + 0.6, 24), rim); orient(h, w); g.add(h);
  }
  // Ground at the lowest point, so it is visible what touches the floor.
  const bb = new THREE.Box3().setFromObject(g), size = bb.getSize(new THREE.Vector3());
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(size.x * 3, size.z * 3), new THREE.MeshLambertMaterial({color: 0x9aa3ad, transparent: true, opacity: 0.25}));
  floor.rotation.x = -Math.PI / 2; floor.position.set(bb.min.x + size.x / 2, bb.min.y - 0.05, bb.min.z + size.z / 2); g.add(floor);
  return g;
}
function orient(mesh, a) {  // a = [x, centre height, y, span, axis]; LEGO y -> three z
  if (a[4] === "y") { mesh.rotation.x = Math.PI / 2; mesh.position.set(a[0], a[1], a[2] + a[3] / 2); }
  else { mesh.rotation.z = Math.PI / 2; mesh.position.set(a[0] + a[3] / 2, a[1], a[2]); }
}
function show(i, keepView) {
  current = i;
  if (group) scene.remove(group);
  group = build(MODELS[i]); scene.add(group);
  document.querySelectorAll("#models button").forEach((b, j) => b.setAttribute("aria-pressed", j === i));
  const s = MODELS[i].scene;
  document.getElementById("facts").innerHTML = MODELS[i].facts.map(f => `<span>${f[0]} <b>${f[1]}</b></span>`).join("");
  if (!keepView) view("three");
}
function view(kind) {
  const bb = new THREE.Box3().setFromObject(group), c = bb.getCenter(new THREE.Vector3()), size = bb.getSize(new THREE.Vector3());
  const d = Math.max(size.x, size.y, size.z) * 2.1;
  const dir = kind === "side" ? [0, 0.08, 1] : kind === "under" ? [0.6, -0.9, 0.8] : [-1.1, 0.75, 1.2];
  camera.position.set(c.x + dir[0] * d, c.y + dir[1] * d, c.z + dir[2] * d);
  controls.target.copy(c); controls.update();
}
MODELS.forEach((m, i) => { const b = document.createElement("button"); b.type = "button"; b.textContent = m.name; b.onclick = () => show(i); document.getElementById("models").appendChild(b); });
["hl", "xray"].forEach(id => document.getElementById(id).onchange = () => show(current, true));
document.getElementById("side").onclick = () => view("side");
document.getElementById("three").onclick = () => view("three");
document.getElementById("under").onclick = () => view("under");
function bg() { scene.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue("--stage").trim() || "#e3e7eb"); }
function resize() { const w = stage.clientWidth, h = stage.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(stage);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", bg);
new MutationObserver(bg).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
bg(); show(0); resize();
(function loop() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); })();
</script>
"""


def page(models: list[tuple[str, list[dict], list[tuple[str, str]]]]) -> str:
    data = [{"name": n, "scene": scene(parts), "facts": facts} for n, parts, facts in models]
    return PAGE.replace("__MODELS__", json.dumps(data, separators=(",", ":")))


def facts_for(parts: list[dict]) -> list[tuple[str, str]]:
    solid = [p for p in parts if p.get("type") in HEIGHT]
    xs = [v for p in solid for v in (p["x"], p["x"] + p["w"])]
    ys = [v for p in solid for v in (p["y"], p["y"] + p["l"])]
    n = lambda t: sum(p.get("type") == t for p in parts)  # noqa: E731
    return [("parts", str(len(parts))), ("length", f"{max(xs) - min(xs)} studs"), ("width", f"{max(ys) - min(ys)} studs"),
            ("Technic bricks", str(n("technic_brick"))), ("axles", str(n("axle"))), ("wheels", str(n("wheel")))]


if __name__ == "__main__":
    out = Path(sys.argv[1])
    models = []
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        parts = json.loads(Path(path).read_text())
        models.append((name, parts, facts_for(parts)))
    out.write_text(page(models))
    print(out, f"{out.stat().st_size / 1e6:.2f} MB")
