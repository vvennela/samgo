"""Interactive 3D page for SAMgo builds: rotate, zoom, compare. Used whenever a model is shared.

Parts are drawn in millimetres at true LEGO scale (stud 8 mm, plate 3.2 mm), with real depth (WebGL), so wheels
and the body occlude each other correctly (matplotlib's 3D renders do not). Axles and wheels sit where the
function verifier puts them: on stud edges at Technic hole height.

    uv run python -m samgo.viewer3d out.html "Glimmer=runs/agent/running_gear/camry_parts.json" \
        "Algorithm=runs/s6/camry/build.json"
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from .kinematics import _sides, analyse, sweep, wheel_axis_matches
from .parts import COLORS
from .technic import (BEAM_HEIGHT, DEFAULT_WHEEL, HOLE_HEIGHT, LENGTH, PIN_RADIUS, PLATE, STUD, TUBE_RADIUS, WHEELS,
                      extent, wheel_studs)

# Highlighted as mechanism parts (with axles and wheels): what holds and turns the moving parts.
MECHANISM_PARTS = {"technic_brick", "hinge_base", "hinge_top", "turntable_base", "turntable_top", "bar_plate",
                   "clip_plate", "click_a", "click_b", "car_door"}
HEIGHT = {"plate": 1, "brick": 3, "technic_brick": 3, "tile": 1, "slope": 3, "invslope": 3, "curve": 2,
          "roundbrick": 3, "roundplate": 1, "arch": 3, "hinge_base": 1, "hinge_top": 1, "car_door": 3, "bar_plate": 1, "clip_plate": 1, "click_a": 1,
          "click_b": 1, "turntable_base": 1, "turntable_top": 1}
EXTRA = {"trans_clear": (70, 110, 150), "trans_black": (45, 60, 75)}
SHAPE = {"slope": 1, "curve": 2, "invslope": 3, "roundbrick": 4, "roundplate": 4}  # 0 = box
# Usual colours of the Technic connectors (LDraw/BrickLink): 2780 black, 3673 gray, 6558 and 43093 blue.
PIN_COLOR = {(2, True): "black", (2, False): "light_gray", (3, True): "blue"}
HOLE = "#16191c"


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(rgb)


def _rgb(p: dict, default: str = "light_gray") -> str:
    c = p.get("color", default)
    return _hex(COLORS.get(c, (None, EXTRA.get(c, (150, 150, 150))))[1])


def _technic(p: dict, boxes: list, cyls: list) -> None:
    """Liftarms as a bar with a rounded end and a dark hole at every hole; cross blocks as their block; pins,
    joiners and bushes as cylinders on their axis line."""
    t, col = p["type"], _rgb(p)
    h = p["z"] * PLATE + HOLE_HEIGHT
    if t == "liftarm":
        lo, hi = extent(p)
        along = 0 if p["axis"] == "x" else 1
        lo, hi = list(lo), list(hi)
        lo[along], hi[along] = lo[along] + STUD / 2, hi[along] - STUD / 2   # bar between the end hole centres
        boxes.append([lo[0], lo[2], lo[1], hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1], col, 1, 0, "", 0])
        across = "y" if p["axis"] == "x" else "x"
        for i in range(p["length"]):
            e = p[p["axis"]] + i
            x, y = (e, p["y"]) if p["axis"] == "x" else (p["x"], e)
            cyls.append([x * STUD, h, y * STUD, STUD, across, BEAM_HEIGHT / 2, col])
            dx, dy = (0, 0.15) if across == "y" else (0.15, 0)  # a little longer than the beam, so it shows
            cyls.append([x * STUD - dx, h, y * STUD - dy, STUD + 0.3, across, PIN_RADIUS, HOLE])
        return
    if t == "propeller":  # hub (round 2x2 on the axle's end), two crossed 2-wide plates on its studs, a round cap
        r = p["blade"] * STUD / 2
        e, s0 = (p["x"], p["y"]) if p["axis"] == "y" else (p["y"], p["x"])
        a0 = s0 * STUD  # the axle's end: hub just before it, blades and cap after
        for k, (wide, tall) in enumerate(((r, STUD), (STUD, r))):  # (half-span across, half-span up)
            if p["axis"] == "y":
                boxes.append([e * STUD - wide, h - tall, a0 + k * PLATE, 2 * wide, 2 * tall, PLATE, col, 1, 0, "", 0])
            else:
                boxes.append([a0 + k * PLATE, h - tall, e * STUD - wide, PLATE, 2 * tall, 2 * wide, col, 1, 0, "", 0])
        for a in (a0 - PLATE, a0 + 2 * PLATE):  # hub and cap: round 2x2 plates
            x, y = (e * STUD, a) if p["axis"] == "y" else (a, e * STUD)
            cyls.append([x, h, y, PLATE, p["axis"], STUD, col])
        return
    if t == "cross_block":
        lo, hi = extent(p)
        boxes.append([lo[0], lo[2], lo[1], hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1], col, 1, 0, "", 0])
        return
    start = p["z"] * PLATE if p["axis"] == "z" else h
    if t == "pin":
        n, fr = p.get("length", 2), bool(p.get("friction", True))
        cyls.append([p["x"] * STUD, start, p["y"] * STUD, n * STUD, p["axis"], PIN_RADIUS, _rgb(p, PIN_COLOR[(n, fr)])])
    elif t == "axle_pin":
        cyls.append([p["x"] * STUD, start, p["y"] * STUD, 2 * STUD, p["axis"], PIN_RADIUS, _rgb(p, "blue")])
    else:  # axle joiner, half bush: a tube round the axle
        cyls.append([p["x"] * STUD, start, p["y"] * STUD, LENGTH[t] * STUD, p["axis"], TUBE_RADIUS, col])


def scene(parts: list[dict]) -> dict:
    """Boxes, axles and wheels in mm, y up (LEGO z), grouped for the page; Technic connectors as boxes and
    cylinders ([x, start height, y, span, axis, radius, colour])."""
    boxes, axles, wheels, cyls = [], [], [], []
    for k, p in enumerate(parts):
        sizes = [len(boxes), len(axles), len(wheels), len(cyls)]
        _part(p, boxes, axles, wheels, cyls)
        for lst, n in zip((boxes, axles, wheels, cyls), sizes):
            for rec in lst[n:]:
                rec.append(k)  # last field: the part it draws, so the page can move it with its joint
    return {"boxes": boxes, "axles": axles, "wheels": wheels, "cyls": cyls}


def _part(p: dict, boxes: list, axles: list, wheels: list, cyls: list) -> None:
    """One part's records (see scene)."""
    t = p.get("type")
    if t in ("liftarm", "cross_block", "pin", "axle_pin", "axle_joiner", "half_bush", "propeller"):
        _technic(p, boxes, cyls)
    elif t in HEIGHT:
        rgb = COLORS.get(p.get("color", "light_gray"), (None, EXTRA.get(p.get("color"), (150, 150, 150))))[1]
        boxes.append([p["x"] * STUD, p["z"] * PLATE, p["y"] * STUD, p["w"] * STUD, HEIGHT[t] * PLATE,
                      p["l"] * STUD, _hex(rgb), 1 if t in MECHANISM_PARTS else 0, SHAPE.get(t, 0),
                      p.get("facing") or "", 1 if str(p.get("color", "")).startswith("trans") else 0])
        if t == "technic_brick":  # its round holes, where axles pass through (the checker's own hole geometry)
            from .technic import _holes

            for axis, edge, z, s0, _ in _holes(p):
                x, y = (edge, s0 - 0.15) if axis == "y" else (s0 - 0.15, edge)
                cyls.append([x, z, y, STUD + 0.3, axis, PIN_RADIUS + 0.6, HOLE])  # a ring round the axle
    elif t in ("axle", "wheel"):
        size = p.get("size", DEFAULT_WHEEL)
        span = (p["length"] if t == "axle" else wheel_studs(size)) * STUD
        centre_h = p["z"] * PLATE + HOLE_HEIGHT
        rec = [p["x"] * STUD, centre_h, p["y"] * STUD, span, p.get("axis", "y")]
        if t == "wheel":
            rec.append(WHEELS[size][0] / 2)       # radius
            rec.append(WHEELS[size][1])           # tyre width, mm
        (axles if t == "axle" else wheels).append(rec)


SPIN_PARTS = {"wheel", "axle", "half_bush", "axle_joiner", "pin", "axle_pin"}  # a wheel with its axle and hub
ROTOR_ARMS = {"liftarm", "cross_block", "propeller"}  # locked on a turning axle: a fan's blades, a propeller
SWINGS = (360, 180, 120, 90, 60, 30, 15)


def motions(parts: list[dict], budget: float = 20.0) -> list[dict]:
    """The build's joints for the page's Simulate mode, outermost first (largest moving set first).

    Each: {"moving": part indices that turn, "point"/"axis": the pivot in the page's mm y-up frame, "spin": a
    wheel making a full turn, "degrees": sampled range (signed, in the page's frame)}. What
    turns is the smaller side of the joint with everything hanging off it (kinematics.requirement_met's rule).
    The swing range is the largest of SWINGS clear in one direction (kinematics.sweep); after `budget` seconds
    unchecked joints are omitted. These are sampled previews, not continuous-motion certificates."""
    t0 = time.monotonic()
    try:
        bodies, joints = analyse(parts)
    except Exception:
        return []
    floor = min((p.get("z", 0) for p in parts), default=0)
    low = {i for i, p in enumerate(parts) if p.get("z", 0) == floor}
    out = []
    for k, (_, _, _, point, axis, partners) in enumerate(joints):
        sides = _sides(bodies, joints, k)
        if sides is None:
            continue
        # On a tie the side resting lowest (on the ground) stays put.
        small = min(sides, key=lambda g: (sum(len(bodies[r]) for r in g), any(i in low for r in g for i in bodies[r])))
        moving = sorted(i for r in small for i in bodies[r])
        types = {parts[i].get("type") for i in moving}
        # A wheel set, or a rotor (arms locked on an axle), makes full turns; anything else swings.
        spin = ((("wheel" in types) or ("axle" in types and types & ROTOR_ARMS))
                and types <= SPIN_PARTS | ROTOR_ARMS | {"bush"}
                and all(wheel_axis_matches(parts[i], point, axis)
                        for i in moving if parts[i].get("type") == "wheel"))
        # blades on a turntable's turning half (an upright rotor) also make full turns
        spin = spin or bool(types & {"turntable_top", "turntable_base"}
                            and types <= {"turntable_top", "turntable_base", "plate", "tile"}
                            and abs(float(np.asarray(axis, float)[2])) > 0.9)
        deg = next((sign * d for d in ((360,) if spin else SWINGS) for sign in (1, -1)
                    if time.monotonic() - t0 < budget
                    and sweep(parts, moving, point, axis, sign * d, ignore=partners, wheels=True) is None), None)
        if deg is None:
            continue  # blocked or unchecked within the budget: no invented animation range
        p, a = np.asarray(point, float), np.asarray(axis, float)
        # LEGO (x, y, z up) -> page (x, z, y): a mirror, so a turn of +d about axis a is -d about the mapped axis.
        out.append({"moving": moving, "point": [round(float(p[0]), 3), round(float(p[2]), 3), round(float(p[1]), 3)],
                    "axis": [float(a[0]), float(a[2]), float(a[1])], "spin": spin, "degrees": -float(deg), "validation": "sampled_clearance"})
    out.sort(key=lambda j: -len(j["moving"]))
    return out


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
  <label><input type="checkbox" id="hl"> Highlight mechanisms (Technic bricks, axles, wheels, hinges, turntables)</label>
  <label><input type="checkbox" id="xray"> See-through body</label>
  <button type="button" id="sim" aria-pressed="false" hidden>Preview motion (sampled)</button>
  <button type="button" id="buildit">Build it</button>
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
// Simulate: each part rides in the group of the joints that move it (outermost first); per frame only those
// groups' matrices change. With Simulate off they stay identity, so the static model is untouched.
let movers = [], simOn = MODELS.map(() => false), simStart = 0;
// Build it: every part has its step in the build order (m.order[part index]); parts appear step by step, the newest
// dropping in from above. Instanced groups are sorted by step, so showing a step is raising the instance count.
let buildables = [], buildOn = false, buildStart = 0;
const stepOf = (m, rec) => (m.order && m.order[rec[rec.length - 1]] != null) ? m.order[rec[rec.length - 1]] : 0;
const dirOf = (m, rec) => (m.dirs && m.dirs[rec[rec.length - 1]]) || [0, 1, 0];
// Accelerating schedule: each step a little shorter than the one before, until the last pieces snap together.
function schedule(steps) {
  const start = [0], dur = [];
  let d = 0.7;
  for (let k = 0; k < steps; k++) { dur.push(d); start.push(start[k] + d); d = Math.max(0.03, d * 0.93); }
  return {start, dur};
}

function build(m) {
  const g = new THREE.Group();
  const chainOf = {}, holders = {"": g};
  (m.joints || []).forEach((j, n) => j.moving.forEach(i => { chainOf[i] = i in chainOf ? chainOf[i] + "," + n : String(n); }));
  movers = []; buildables = [];
  const holder = rec => {
    const key = chainOf[rec[rec.length - 1]] || "";
    if (!holders[key]) {
      const h = new THREE.Group(); h.matrixAutoUpdate = false; h.userData.chain = key.split(",").map(Number);
      holders[key] = h; movers.push(h); g.add(h);
    }
    return holders[key];
  };
  const hl = document.getElementById("hl").checked, xray = document.getElementById("xray").checked;
  const geo = {0: new THREE.BoxGeometry(1, 1, 1), 1: wedge(false), 2: wedge(true), 3: wedge(false, true),
               4: new THREE.CylinderGeometry(0.5, 0.5, 1, 20)};
  const groups = {};
  for (const b of m.scene.boxes) {
    const col = (hl && b[7]) ? "#ffb000" : b[6];
    (groups[[col, b[8], b[10], chainOf[b[11]] || ""].join("|")] ||= []).push(b);
  }
  for (const [key, list] of Object.entries(groups)) {
    const [col, shape, glass] = key.split("|");
    const technic = hl && col === "#ffb000";
    const see = glass === "1" ? 0.62 : (xray && !technic ? 0.18 : 1);
    const mat = new THREE.MeshLambertMaterial({color: col, transparent: see < 1, opacity: see});
    list.sort((a, b) => stepOf(m, a) - stepOf(m, b));
    const inst = new THREE.InstancedMesh(geo[shape], mat, list.length);
    const mtx = new THREE.Matrix4(), rot = new THREE.Matrix4(), scl = new THREE.Matrix4();
    list.forEach((b, i) => {
      // Wedges are modeled rising toward -x (high side) and sloping down to +x; turn them to their facing.
      const turn = {"+x": 0, "-x": Math.PI, "+y": -Math.PI / 2, "-y": Math.PI / 2}[b[9]] || 0;
      const turned = Math.abs(Math.sin(turn)) > 0.5;
      scl.makeScale(turned ? b[5] - 0.2 : b[3] - 0.2, b[4] - 0.1, turned ? b[3] - 0.2 : b[5] - 0.2);
      rot.makeRotationY(turn);
      mtx.multiplyMatrices(rot, scl);
      mtx.setPosition(b[0] + b[3] / 2, b[1] + b[4] / 2, b[2] + b[5] / 2);
      inst.setMatrixAt(i, mtx);
    });
    inst.userData.steps = list.map(b => stepOf(m, b));
    inst.userData.dirs = list.map(b => dirOf(m, b));
    inst.userData.base = list.map((b, i) => { const q = new THREE.Matrix4(); inst.getMatrixAt(i, q); return q; });
    buildables.push(inst);
    holder(list[0]).add(inst);
  }
  const axleMat = new THREE.MeshLambertMaterial({color: hl ? 0xffb000 : 0x7a7f86});
  for (const a of m.scene.axles) {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(2.4, 2.4, a[3], 16), axleMat);
    orient(c, a); holder(a).add(c); c.userData.step = stepOf(m, a); c.userData.dir = dirOf(m, a); buildables.push(c);
  }
  for (const c of m.scene.cyls || []) {
    const hole = c[6] === "#16191c";
    const mat = new THREE.MeshLambertMaterial({color: (hl && !hole) ? 0xffb000 : c[6]});
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(c[5], c[5], c[3], 20), mat);
    orient(mesh, c); holder(c).add(mesh); mesh.userData.step = stepOf(m, c); mesh.userData.dir = dirOf(m, c); buildables.push(mesh);
  }
  const spokeMat = new THREE.MeshLambertMaterial({color: 0x4a4f57});
  const tyre = new THREE.MeshLambertMaterial({color: 0x1b1b1b}), rim = new THREE.MeshLambertMaterial({color: hl ? 0xffb000 : 0xc9ccd1});
  for (const w of m.scene.wheels) {
    const r = w[5], tw = w[6];
    const t = new THREE.Mesh(new THREE.CylinderGeometry(r, r, tw, 40), tyre); orient(t, w); holder(w).add(t);
    const h = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.6, r * 0.6, tw + 0.6, 24), rim); orient(h, w); holder(w).add(h);
    // Spokes across the hub: a plain cylinder looks the same at every angle, so without them a spinning wheel
    // shows no motion at all.
    const spokes = new THREE.Group();
    for (const turn of [0, Math.PI / 2]) {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(r * 1.15, tw + 1.2, r * 0.16), spokeMat);
      bar.rotation.y = turn; spokes.add(bar);
    }
    orient(spokes, w); holder(w).add(spokes);
    for (const o of [t, h, spokes]) { o.userData.step = stepOf(m, w); o.userData.dir = dirOf(m, w); buildables.push(o); }
  }
  // Ground at the lowest point, so it is visible what touches the floor.
  const bb = bounds(g), size = bb.getSize(new THREE.Vector3());
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(size.x * 1.6, size.z * 1.6), new THREE.MeshLambertMaterial({color: 0x9aa3ad, transparent: true, opacity: 0.25}));
  floor.rotation.x = -Math.PI / 2; floor.position.set(bb.min.x + size.x / 2, bb.min.y - 0.05, bb.min.z + size.z / 2); g.add(floor);
  return g;
}
function wedge(curved, inverted) {
  // A unit block whose top falls from the -x edge to the +x edge (a slope facing +x); curved: a quarter round.
  const pts = [], n = curved ? 8 : 1;
  for (let i = 0; i <= n; i++) {
    const t = i / n, x = -0.5 + t, y = curved ? 0.5 * Math.cos(t * Math.PI / 2) * 2 - 0.5 : 0.5 - t;
    pts.push([x, inverted ? -y : y]);
  }
  const shape = new THREE.Shape();
  shape.moveTo(-0.5, inverted ? 0.5 : -0.5);
  for (const [x, y] of pts) shape.lineTo(x, y);
  shape.lineTo(0.5, inverted ? 0.5 : -0.5);
  const g = new THREE.ExtrudeGeometry(shape, {depth: 1, bevelEnabled: false});
  g.translate(0, 0, -0.5);
  return g;
}
function orient(mesh, a) {  // a = [x, centre height, y, span, axis]; LEGO y -> three z; axis z: from that height up
  if (a[4] === "z") { mesh.position.set(a[0], a[1] + a[3] / 2, a[2]); }
  else if (a[4] === "y") { mesh.rotation.x = Math.PI / 2; mesh.position.set(a[0], a[1], a[2] + a[3] / 2); }
  else { mesh.rotation.z = Math.PI / 2; mesh.position.set(a[0] + a[3] / 2, a[1], a[2]); }
}
function show(i, keepView) {
  current = i;
  if (group) scene.remove(group);
  group = build(MODELS[i]); scene.add(group);
  document.querySelectorAll("#models button").forEach((b, j) => b.setAttribute("aria-pressed", j === i));
  const s = MODELS[i].scene;
  document.getElementById("facts").innerHTML = MODELS[i].facts.map(f => `<span>${f[0]} <b>${f[1]}</b></span>`).join("");
  const n = (MODELS[i].joints || []).length, sim = document.getElementById("sim");
  sim.hidden = n === 0;
  sim.setAttribute("aria-pressed", simOn[i]);
  const spins = (MODELS[i].joints || []).filter(j => j.spin).length;
  sim.title = `${spins} spinning, ${n - spins} swinging joint(s); sampled clearance, one at a time`;
  if (!keepView) view("three");
}
const jm = new THREE.Matrix4(), jr = new THREE.Matrix4(), jt = new THREE.Matrix4(), jv = new THREE.Vector3();
function jointMatrix(j, t, out) {  // turn about the pivot line: T(p) R(axis, angle) T(-p)
  let a;
  a = j.degrees * Math.PI / 180 * (1 - Math.cos(2 * Math.PI * t / 6)) / 2;
  jr.makeRotationAxis(jv.set(j.axis[0], j.axis[1], j.axis[2]).normalize(), a);
  out.makeTranslation(j.point[0], j.point[1], j.point[2]).multiply(jr);
  return out.multiply(jt.makeTranslation(-j.point[0], -j.point[1], -j.point[2]));
}
function buildFrame() {
  if (!buildOn) return;
  const m = MODELS[current], steps = (m.order || [0]).reduce((a, b) => Math.max(a, b), 0) + 1;
  const sch = m._sch || (m._sch = schedule(steps)), drop = 60;
  const t = (performance.now() - buildStart) / 1000;
  let cur = 0; while (cur < steps && sch.start[cur + 1] <= t) cur++;
  const f = cur < steps ? (t - sch.start[cur]) / sch.dur[cur] : 1;
  const ease = 1 - Math.pow(1 - Math.min(1, f / 0.8), 3);  // arrives fast, settles
  const done = cur >= steps;
  const q = new THREE.Matrix4(), mv = new THREE.Matrix4();
  for (const o of buildables) {
    if (o.isInstancedMesh) {
      const st = o.userData.steps, base = o.userData.base, ds = o.userData.dirs;
      let n = 0; while (n < st.length && (done || st[n] <= cur)) n++;
      o.count = n;
      for (let i = 0; i < n; i++) {
        if (!done && st[i] === cur) {  // along its proved-clear direction (formal/AnimationPath.lean)
          const d = ds[i], k = drop * (1 - ease);
          mv.makeTranslation(d[0] * k, d[1] * k, d[2] * k); q.multiplyMatrices(mv, base[i]); o.setMatrixAt(i, q);
        } else o.setMatrixAt(i, base[i]);
      }
      o.instanceMatrix.needsUpdate = true;
    } else {
      if (o.userData.p0 === undefined) o.userData.p0 = o.position.clone();
      o.visible = done || o.userData.step <= cur;
      const d = o.userData.dir || [0, 1, 0], k = (!done && o.userData.step === cur) ? drop * (1 - ease) : 0;
      o.position.set(o.userData.p0.x + d[0] * k, o.userData.p0.y + d[1] * k, o.userData.p0.z + d[2] * k);
    }
  }
  if (done) buildOn = false;
}
function animate() {
  const on = simOn[current], joints = MODELS[current].joints || [];
  const t = (performance.now() - simStart) / 1000, cache = {};
  const active = Math.floor(t / 6) % joints.length; // every preview starts and ends at rest
  for (const h of movers) {
    if (!on && h.userData.rest) continue;  // off: back at rest once, then left alone
    h.userData.rest = !on;
    h.matrix.identity();
    if (on && h.userData.chain.includes(active)) h.matrix.multiply(cache[active] ||= jointMatrix(joints[active], t % 6, new THREE.Matrix4()));
    h.matrixWorldNeedsUpdate = true;
  }
}
function bounds(obj) {  // Box3.setFromObject sees one unit box per InstancedMesh (three r128); count every instance
  const bb = new THREE.Box3(), m = new THREE.Matrix4(), one = new THREE.Box3();
  obj.updateMatrixWorld(true);
  obj.traverse(o => {
    if (!o.isMesh || !o.visible) return;
    if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
    if (o.isInstancedMesh) for (let i = 0; i < o.count; i++) {
      o.getMatrixAt(i, m); bb.union(one.copy(o.geometry.boundingBox).applyMatrix4(m.premultiply(o.matrixWorld)));
    } else bb.union(one.copy(o.geometry.boundingBox).applyMatrix4(o.matrixWorld));
  });
  return bb;
}
function view(kind) {
  const bb = bounds(group), c = bb.getCenter(new THREE.Vector3()), size = bb.getSize(new THREE.Vector3());
  const d = Math.max(size.x, size.y, size.z) * 1.15;
  const dir = kind === "side" ? [0, 0.08, 1] : kind === "under" ? [0.6, -0.9, 0.8] : [-1.1, 0.75, 1.2];
  camera.position.set(c.x + dir[0] * d, c.y + dir[1] * d, c.z + dir[2] * d);
  controls.target.copy(c); controls.update();
}
MODELS.forEach((m, i) => { const b = document.createElement("button"); b.type = "button"; b.textContent = m.name; b.onclick = () => show(i); document.getElementById("models").appendChild(b); });
["hl", "xray"].forEach(id => document.getElementById(id).onchange = () => show(current, true));
document.getElementById("buildit").onclick = () => { buildOn = true; buildStart = performance.now(); };
document.getElementById("sim").onclick = () => {
  simOn[current] = !simOn[current]; simStart = performance.now();
  document.getElementById("sim").setAttribute("aria-pressed", simOn[current]);
};
document.getElementById("side").onclick = () => view("side");
document.getElementById("three").onclick = () => view("three");
document.getElementById("under").onclick = () => view("under");
function bg() { scene.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue("--stage").trim() || "#e3e7eb"); }
function resize() { const w = stage.clientWidth, h = stage.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(stage);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", bg);
new MutationObserver(bg).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
bg(); show(0); resize();
// Recording (scripts/record_video.py): one exact frame at time t, every joint moving at once (spins turn on, swings
// open and close), the camera orbiting; o.build = seconds into the build animation. The live loop stands down.
window.RECORDING = false;
function showUpTo(k) {
  for (const o of buildables) {
    if (o.isInstancedMesh) {
      const st = o.userData.steps, base = o.userData.base;
      let n = 0; while (n < st.length && st[n] <= k) n++;
      o.count = n;
      for (let i = 0; i < n; i++) o.setMatrixAt(i, base[i]);
      o.instanceMatrix.needsUpdate = true;
    } else {
      if (o.userData.p0 === undefined) o.userData.p0 = o.position.clone();
      o.visible = o.userData.step <= k; o.position.copy(o.userData.p0);
    }
  }
}
window.renderAt = (i, t, o = {}) => {
  window.RECORDING = true;
  if (current !== i || !group) { show(i, true); MODELS[i]._bb = null; }
  const m = MODELS[i], joints = m.joints || [];
  if (!m._bb) { buildOn = false; buildFrame(); m._bb = bounds(group); }
  const mats = joints.map(j => {
    const a = j.spin ? Math.sign(j.degrees || 1) * 2 * Math.PI * t / (o.spin || 2.5)
                     : j.degrees * Math.PI / 180 * (1 - Math.cos(2 * Math.PI * t / (o.swing || 4))) / 2;
    jr.makeRotationAxis(jv.set(j.axis[0], j.axis[1], j.axis[2]).normalize(), a);
    const out = new THREE.Matrix4().makeTranslation(j.point[0], j.point[1], j.point[2]).multiply(jr);
    return out.multiply(new THREE.Matrix4().makeTranslation(-j.point[0], -j.point[1], -j.point[2]));
  });
  for (const h of movers) {
    h.matrix.identity();
    if (o.move !== false) for (const n of h.userData.chain) if (mats[n]) h.matrix.multiply(mats[n]);
    h.matrixWorldNeedsUpdate = true; h.userData.rest = false;
  }
  if (o.build != null) { buildOn = true; buildStart = performance.now() - o.build * 1000; buildFrame(); }
  else if (buildOn) { buildOn = false; }
  if (o.step != null) showUpTo(o.step);  // instructions: everything placed up to step o.step, at rest
  let bb = m._bb;
  if (o.step != null && o.fit) {  // frame what is built so far (not less than o.fit of the whole model)
    if (o.fitTo != null) showUpTo(o.fitTo);  // frame for another step (so before/after pictures share a camera)
    const part = bounds(group);
    if (o.fitTo != null) showUpTo(o.step);
    const full = m._bb.getSize(new THREE.Vector3()), ctr = part.getCenter(new THREE.Vector3());
    const half = part.getSize(new THREE.Vector3()).max(full.clone().multiplyScalar(o.fit)).multiplyScalar(0.5);
    bb = new THREE.Box3(ctr.clone().sub(half), ctr.clone().add(half));
  }
  const c = bb.getCenter(new THREE.Vector3()), size = bb.getSize(new THREE.Vector3());
  const R = Math.max(size.x, size.y, size.z) * (o.dist || 1.55), ang = (o.az || 0.8) + 2 * Math.PI * t / (o.orbit || 14);
  const flip = o.flip ? -1 : 1;  // a step built upside down: seen from below, the model turned over
  camera.up.set(0, flip, 0);
  camera.position.set(c.x + Math.cos(ang) * R, c.y + flip * R * (o.elev || 0.45), c.z + Math.sin(ang) * R);
  camera.lookAt(c); renderer.render(scene, camera);
  return true;
};
(function loop() {
  if (!window.RECORDING) { controls.update(); buildFrame(); animate(); renderer.render(scene, camera); }
  requestAnimationFrame(loop);
})();
</script>
"""


def build_order(parts: list[dict], manual: dict | None = None) -> list[int]:
    """The step of each part: the certified order (samgo.manual) when there is one, else layers bottom-up, then
    axles and wheels (samgo.instructions.steps)."""
    order = [0] * len(parts)
    if manual and manual.get("passed") and manual.get("steps"):
        for k, s in enumerate(manual["steps"]):
            for i in s["parts"]:
                order[i] = k
        return order
    from .instructions import steps

    index = {id(p): i for i, p in enumerate(parts)}
    try:
        for k, group in enumerate(steps(parts)):
            for p in group:
                order[index[id(p)]] = k
    except Exception:  # noqa: BLE001 - parts the schematic order does not know: by height
        zs = sorted({p.get("z", 0) for p in parts})
        order = [zs.index(p.get("z", 0)) for p in parts]
    return order


def page(models: list[tuple], simulate: bool = True) -> str:
    """models: (name, parts, facts) or (name, parts, facts, manual)."""
    from .animation import order as anim_order
    from .animation import paths as anim_paths

    data = []
    for m in models:
        steps = anim_order(m[1], m[3] if len(m) > 3 else None)
        dirs, _ = anim_paths(m[1], steps)
        data.append({"name": m[0], "scene": scene(m[1]), "facts": m[2], "joints": motions(m[1]) if simulate else [],
                     "order": steps, "dirs": [[d[0], d[2], d[1]] for d in dirs]})  # LEGO (x, y, z) -> page (x, z, y)
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
        manual = Path(path).with_name("manual.json")
        models.append((name, parts, facts_for(parts), json.loads(manual.read_text()) if manual.exists() else None))
    out.write_text(page(models))
    print(out, f"{out.stat().st_size / 1e6:.2f} MB")
