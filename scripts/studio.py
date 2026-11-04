"""SAMgo Studio: a live view of a real pipeline run. It watches the run's output directory and shows each stage the
moment its artifact lands (photo, SAM 3.1 mask and glass, SAM 3D mesh, the LEGO body, Spark's layout, Glimmer's
edits and the checks, the finished model in the 3D viewer), with every live model call in a side panel
(<out>/calls.jsonl, written by scripts/studio_run.py).

    python scripts/studio.py <out dir> [port=8765]
"""
import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>SAMgo Studio</title>
<style>
:root { --bg:#f2f3f5; --panel:#fff; --ink:#1d2329; --muted:#6e7884; --line:#e3e6ea; --red:#d01012; --green:#16a050;
  --blue:#0866ff; --violet:#7c3aed; }
* { box-sizing:border-box; }
html,body { margin:0; height:100%; background:var(--bg); color:var(--ink); font-family:Futura, "Avenir Next", Helvetica, sans-serif; }
header { display:flex; align-items:center; gap:18px; padding:18px 28px; }
header h1 { margin:0; font-size:26px; letter-spacing:.06em; font-weight:500; }
header .run { font:13px Menlo, monospace; color:var(--muted); }
header .stage { margin-left:auto; font-size:15px; letter-spacing:.08em; color:var(--muted); text-transform:uppercase; }
header .dot { width:10px; height:10px; border-radius:50%; background:var(--green); box-shadow:0 0 0 0 rgba(22,160,80,.5); animation:pulse 1.4s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(22,160,80,.45)} 70%{box-shadow:0 0 0 10px rgba(22,160,80,0)} 100%{box-shadow:0 0 0 0 rgba(22,160,80,0)} }
main { display:grid; grid-template-columns: 470px 1fr; gap:18px; padding:0 28px 24px; height:calc(100% - 70px); }
.calls { background:var(--panel); border-radius:16px; padding:16px 18px; overflow:hidden; display:flex; flex-direction:column; }
.calls h2, .card h2 { margin:0 0 10px; font-size:13px; letter-spacing:.14em; color:var(--muted); font-weight:500; }
#calls { overflow:hidden; display:flex; flex-direction:column; gap:10px; }
.call { border:1px solid var(--line); border-radius:12px; padding:10px 12px; animation:in .35s ease-out; }
.call .who { display:flex; align-items:center; gap:8px; font-size:14px; letter-spacing:.06em; }
.call .who b { font-weight:600; }
.call .who .s { margin-left:auto; font:12px Menlo, monospace; color:var(--muted); }
.call .ask, .call .out { font:12px/1.4 Menlo, monospace; color:var(--muted); margin-top:5px; word-break:break-word; }
.call .out { color:var(--ink); }
.spin { width:12px; height:12px; border:2px solid var(--line); border-top-color:var(--ink); border-radius:50%; animation:sp .8s linear infinite; }
@keyframes sp { to { transform:rotate(360deg) } }
@keyframes in { from { opacity:0; transform:translateY(8px) } to { opacity:1; transform:none } }
.c-sam { color:#0866ff } .c-spark { color:var(--violet) } .c-glimmer { color:#0a7cff } .c-sam3d { color:#00897b }
#stages { display:grid; grid-template-columns: repeat(3, 1fr); grid-auto-rows: minmax(0, 1fr); gap:18px; overflow:hidden; }
.card { background:var(--panel); border-radius:16px; padding:14px 16px; display:flex; flex-direction:column; min-height:0; animation:in .45s ease-out; }
.card .body { flex:1; min-height:0; display:flex; align-items:center; justify-content:center; overflow:hidden; border-radius:10px; position:relative; }
.card img { max-width:100%; max-height:100%; object-fit:contain; border-radius:8px; }
.card .stack { position:relative; width:100%; height:100%; }
.card .stack img { position:absolute; inset:0; margin:auto; }
.card pre { margin:0; width:100%; height:100%; overflow:hidden; font:12px/1.45 Menlo, monospace; color:var(--ink); white-space:pre-wrap; }
.ok { color:var(--green) } .bad { color:var(--red) }
.wide { grid-column: span 3; }
.final { grid-column: 1 / span 3; grid-row: 1 / span 2; }
iframe { width:100%; height:100%; border:0; border-radius:10px; background:#e3e7eb; }
canvas.mesh { width:100%; height:100%; display:block; }
</style></head><body>
<header><div class="dot"></div><h1>SAMgo Studio</h1><span class="run" id="run"></span><span class="stage" id="stage">waiting for a run</span></header>
<main><section class="calls"><h2>LIVE MODEL CALLS</h2><div id="calls"></div></section><section id="stages"></section></main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script>
const shown = {}, cls = m => m.includes("SAM 3D") ? "c-sam3d" : m.includes("SAM") ? "c-sam" : m.includes("Spark") ? "c-spark" : m.includes("Glimmer") ? "c-glimmer" : "";
function card(id, title, html, extra) {
  if (shown[id]) return shown[id];
  const c = document.createElement("div"); c.className = "card " + (extra || ""); c.id = "card-" + id;
  c.innerHTML = `<h2>${title}</h2><div class="body">${html}</div>`;
  document.getElementById("stages").appendChild(c); shown[id] = c; return c;
}
function meshView(url, el) {
  const r = new THREE.WebGLRenderer({antialias:true, alpha:true}); el.appendChild(r.domElement); r.domElement.className = "mesh";
  const s = new THREE.Scene(), cam = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
  s.add(new THREE.HemisphereLight(0xffffff, 0x99a0aa, 1.1)); const d = new THREE.DirectionalLight(0xffffff, .6); d.position.set(2,3,2); s.add(d);
  new THREE.GLTFLoader().load(url, g => {
    const o = g.scene; const b = new THREE.Box3().setFromObject(o), c = b.getCenter(new THREE.Vector3()), z = b.getSize(new THREE.Vector3()).length();
    o.position.sub(c); s.add(o); let a = 0.6;
    (function loop() { const w = el.clientWidth, h = el.clientHeight; r.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix();
      a += 0.006; cam.position.set(Math.cos(a) * z * 1.1, z * .35, Math.sin(a) * z * 1.1); cam.lookAt(0, 0, 0); r.render(s, cam); requestAnimationFrame(loop); })();
  });
}
let lastCalls = 0;
async function tick() {
  let st; try { st = await (await fetch("/state")).json(); } catch (e) { return; }
  if (!st.summary) return;
  const S = st.summary, F = st.files;
  document.getElementById("run").textContent = (S.photo || "").split("/").pop() + "  ·  " + (S.prompt || "") + (S.replay ? "  ·  recorded live run, Nov 4" : "");
  document.getElementById("stage").textContent = S.status === "running" ? (S.stage || "").replaceAll("_", " ") : S.status;
  // calls: newest at the top, "start" replaced by its "done"
  if (st.calls.length !== lastCalls) {
    lastCalls = st.calls.length; const box = document.getElementById("calls"); box.innerHTML = "";
    const open = {}, list = [];
    for (const c of st.calls) {
      if (c.ev === "start") { const k = c.model + list.length; open[c.model] = k; list.push({...c, k, pending: true}); }
      else if (c.ev === "done") { const i = list.map(x => x.model).lastIndexOf(c.model); if (i >= 0) Object.assign(list[i], c, {pending: false}); }
    }
    for (const c of list.slice(-9).reverse()) {
      const d = document.createElement("div"); d.className = "call";
      d.innerHTML = `<div class="who">${c.pending ? '<div class="spin"></div>' : '<span class="ok">✓</span>'}<b class="${cls(c.model)}">${c.model}</b>
        <span class="s">${c.pending ? "calling…" : c.secs + " s"}</span></div>` +
        (c.ask ? `<div class="ask">→ ${c.ask.replace(/</g, "&lt;")}</div>` : "") +
        (c.out ? `<div class="out">← ${c.out.replace(/</g, "&lt;")}</div>` : "");
      box.appendChild(d);
    }
  }
  if (F["photo.png"]) card("photo", "THE PHOTO", `<img src="/f/photo.png?${F["photo.png"]}">`);
  if (F["mask.png"]) card("sam", "SAM 3.1  ·  THE CAR AND ITS GLASS", `<div class="stack"><img src="/f/photo.png"><img id="m1" src="/f/mask_tint.png?${F["mask.png"]}"></div>`);
  if (F["glass_mask.png"] && shown.sam && !shown.glass) { shown.glass = 1; const s = shown.sam.querySelector(".stack"); const i = document.createElement("img"); i.src = "/f/glass_tint.png?" + F["glass_mask.png"]; s.appendChild(i); }
  if (F["object.glb"] && !shown.mesh) { const c = card("mesh", "SAM 3D  ·  ONE PHOTO TO A 3D SHAPE", ""); meshView("/f/object.glb", c.querySelector(".body")); }
  if (F["target.png"]) card("body", "THE LEGO BODY  ·  REAL PLATES", `<img src="/f/target.png?${F["target.png"]}">`);
  if (F["layout/side_view.png"]) card("spark", "MUSE SPARK  ·  WHEELS AND DOORS", `<img src="/f/layout/side_view.png?${F["layout/side_view.png"]}">`);
  if (st.glimmer) { const c = card("glimmer", "MUSE GLIMMER  ·  ITS EDITS, CHECKED", "<pre></pre>", "wide"); c.querySelector("pre").innerHTML = st.glimmer; }
  if (F["viewer.html"] && (S.status === "passed" || S.status === "failed" || S.stage === "complete")) {
    if (!shown.final) { document.getElementById("stages").innerHTML = ""; for (const k in shown) delete shown[k];
      card("final", "THE BUILD  ·  " + (S.parts || "") + " REAL PARTS", `<iframe id="viewer" src="/f/viewer.html"></iframe>`, "final"); }
  }
}
setInterval(tick, 300); tick();
</script></body></html>"""


class Handler(SimpleHTTPRequestHandler):
    out: Path = None
    replay: dict = None         # {"t0": wall time the replay started, "start": first call's time, "speed": x}

    def now(self):
        """In replay, the moment of the recorded run being shown; live, None (everything so far)."""
        r = Handler.replay
        if not r:
            return None
        import time
        return r["start"] + (time.time() - r["t0"]) * r["speed"]

    def log_message(self, *a):
        pass

    def run_dir(self):
        s = self.out / "summary.json"
        try:
            return Path(json.loads(s.read_text())["run_dir"]), json.loads(s.read_text())
        except Exception:  # noqa: BLE001
            return None, None

    def send(self, body: bytes, kind: str):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send(PAGE.encode(), "text/html; charset=utf-8")
        run, summary = self.run_dir()
        if self.path == "/state":
            now = self.now()
            calls = []
            log = self.out / "calls.jsonl"
            if log.exists():
                for line in log.read_text().splitlines():
                    try:
                        c = json.loads(line)
                    except ValueError:
                        continue
                    if now is None or c.get("t", 0) <= now:
                        calls.append(c)
            files = {}
            if run and run.exists():
                for p in run.rglob("*"):
                    if p.is_file() and (now is None or p.stat().st_mtime <= now):
                        files[str(p.relative_to(run))] = int(p.stat().st_mtime * 1000)
            if now is not None and summary:
                ended = any(c.get("ev") == "end" for c in calls)
                summary = dict(summary, status=summary.get("status") if ended else "running",
                               stage=STAGE_OF.get((calls[-1].get("model") if calls else ""), "starting"),
                               replay=True)
            if now is not None and not (run / "assembly" / "trace.json").exists() or (
                    now is not None and (run / "assembly" / "trace.json").stat().st_mtime > now):
                return self.send(json.dumps({"summary": summary, "calls": calls, "files": files,
                                             "glimmer": None}).encode(), "application/json")
            return self.send(json.dumps({"summary": summary, "calls": calls, "files": files,
                                         "glimmer": glimmer_view(run)}).encode(), "application/json")
        if self.path.startswith("/f/") and run:
            name = self.path[3:].split("?")[0]
            if name in ("mask_tint.png", "glass_tint.png"):
                return self.send(tint(run, name), "image/png")
            p = (run / name).resolve()
            if run.resolve() in p.parents and p.exists():
                kind = {"png": "image/png", "glb": "model/gltf-binary", "html": "text/html; charset=utf-8",
                        "json": "application/json"}.get(p.suffix[1:], "application/octet-stream")
                return self.send(p.read_bytes(), kind)
        self.send_response(404)
        self.end_headers()


STAGE_OF = {"SAM 3.1": "SAM 3.1 · segmenting", "SAM 3D": "SAM 3D · reconstructing", "Muse Spark": "Muse Spark · reading",
            "Muse Glimmer": "Muse Glimmer · building", "SAMgo": "starting"}


def tint(run: Path, name: str) -> bytes:
    """SAM 3.1's masks as overlays: the car outlined in blue, its glass in magenta."""
    import io
    import numpy as np
    from PIL import Image, ImageFilter
    src = run / ("mask.png" if name == "mask_tint.png" else "glass_mask.png")
    m = np.asarray(Image.open(src).convert("L")) > 127
    rgba = np.zeros(m.shape + (4,), np.uint8)
    if name == "mask_tint.png":
        e = Image.fromarray(m.astype(np.uint8) * 255)
        edge = np.asarray(e.filter(ImageFilter.MaxFilter(9))) ^ np.asarray(e.filter(ImageFilter.MinFilter(9)))
        rgba[edge > 0] = (8, 102, 255, 255)
        rgba[~m] = (240, 242, 245, 150)
    else:
        rgba[m] = (255, 0, 255, 140)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def glimmer_view(run):
    """Glimmer's edits per axle line and the checker's verdict, from the run's own trace."""
    if not run:
        return None
    t = run / "assembly" / "trace.json"
    if not t.exists():
        return None
    try:
        trace = json.loads(t.read_text())
    except ValueError:
        return None
    lines = []
    for s in trace.get("steps", []):
        for a in s.get("attempts", []):
            lines.append(f"<span class='c-glimmer'>Glimmer</span>  axle line at stud {s.get('edge')}, round {a.get('round', 0) + 1}")
            for e in a.get("edits", [])[:10]:
                part = e.get("part", {})
                lines.append(f"  {e.get('op')} {part.get('type')} {part.get('w', '')}x{part.get('l', '')} "
                             f"at ({part.get('x')}, {part.get('y')}, {part.get('z')})")
            ok = a.get("passed")
            issues = "; ".join(str(i) for i in (a.get("issues") or [])[:2])
            lines.append(f"<span class='{'ok' if ok else 'bad'}'>checker  {'✓ rolls, held by studs' if ok else '✗ ' + issues}</span>")
    return "\n".join(lines)


def main():
    out = Path(sys.argv[1]).resolve()
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    Handler.out = out
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"SAMgo Studio on http://127.0.0.1:{port}  watching {out}", flush=True)
    srv.serve_forever()


def serve_in_thread(out: Path, port: int = 8765, replay_speed: float | None = None):
    Handler.out = out.resolve()
    if replay_speed:
        import time
        first = json.loads((out / "calls.jsonl").read_text().splitlines()[0])["t"]
        Handler.replay = {"t0": time.time(), "start": first - 1.0, "speed": replay_speed}
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    main()
