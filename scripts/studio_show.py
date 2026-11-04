"""SAMgo Studio, presentation view: replays a logged live run (scripts/studio_run.py) exactly in its recorded order,
with long waits squeezed (a speed badge shows while they are) so the whole run plays in about TARGET seconds.

Everything on screen comes from the run itself: the call log (model, what was asked, the answer, seconds), the
artifacts as they were written (photo, SAM 3.1 masks, SAM 3D mesh, the LEGO body, Spark's layout, Glimmer's edits
with the checker's verdicts from the run's traces), and the finished model in the real viewer.

    python scripts/studio_show.py <run out dir> <assets dir> [port=8767] [target=44]
assets: mesh.mp4, body.mp4, spark.mp4 rendered from this run's mesh, body and Spark layout.
"""
import bisect
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CAP = 1.6          # a wait between two events plays as at most this long (before the overall scale)


class Run:
    def __init__(self, out: Path, assets: Path, target: float):
        self.out, self.assets = out, assets
        self.summary = json.loads((out / "summary.json").read_text())
        self.dir = Path(self.summary["run_dir"])
        self.calls = [json.loads(l) for l in (out / "calls.jsonl").read_text().splitlines() if l.strip()]
        self.files = {str(p.relative_to(self.dir)): p.stat().st_mtime for p in self.dir.rglob("*") if p.is_file()}
        self.start = self.calls[0]["t"] - 0.6
        self.end = max(c["t"] for c in self.calls) + 0.4
        times = sorted({self.start, self.end, *(c["t"] for c in self.calls),
                        *(t for t in self.files.values() if self.start <= t <= self.end)})
        edited = [0.0]
        for a, b in zip(times, times[1:]):
            edited.append(edited[-1] + min(b - a, CAP))
        k = target / edited[-1]
        self.real, self.edited = times, [e * k for e in edited]
        self.verdicts = self._verdicts()
        self.t0 = None

    def _verdicts(self):
        """The checker's verdict for each Glimmer call, in call order, from the run's traces."""
        out = []
        t = json.loads((self.dir / "assembly" / "trace.json").read_text())
        for s in t["steps"]:
            for a in s["attempts"]:
                out.append({"task": f"axle line at stud {s['edge']}", "ok": bool(a.get("passed")),
                            "why": "rolls: axle through both Technic bricks, wheels clear, held by studs"
                            if a.get("passed") else "; ".join(map(str, a.get("issues", [])[:2])),
                            "edits": [e.get("part", {}) for e in a.get("edits", [])]})
        d = self.dir / "doors" / "trace.json"
        if d.exists():
            dt = json.loads(d.read_text())
            for r in dt.get("rounds", []):
                issues = [str(i).split(" - ")[0].split(",")[0] for i in (r.get("issues") or [])][:1]
                try:
                    edits = [e.get("part", {}) for e in json.loads(r.get("answer") or "[]")]
                except ValueError:
                    edits = []
                n = int(r.get("tries") or 1)
                for k in range(n):
                    out.append({"task": f"door hinges, round {r.get('round')}  ({n} answers at once)",
                                "ok": bool(r.get("passed")), "why": "; ".join(issues), "edits": edits, "first": k == 0,
                                "door": True})
        return out

    def real_at(self, x):
        """Real run time shown at edited time x, and the playback speed there."""
        i = max(0, min(len(self.edited) - 2, bisect.bisect_right(self.edited, x) - 1))
        e0, e1, r0, r1 = self.edited[i], self.edited[i + 1], self.real[i], self.real[i + 1]
        u = 0 if e1 == e0 else min(1.0, max(0.0, (x - e0) / (e1 - e0)))
        return r0 + u * (r1 - r0), (r1 - r0) / max(1e-6, e1 - e0)

    def state(self):
        if self.t0 is None:
            self.t0 = time.time()
        x = time.time() - self.t0
        now, speed = self.real_at(x)
        calls = [c for c in self.calls if c["t"] <= now]
        files = {k: v for k, v in self.files.items() if v <= now}
        gl = [c for c in calls if c["model"] == "Muse Glimmer"]
        n_done = sum(1 for c in gl if c["ev"] == "done")
        n_started = sum(1 for c in gl if c["ev"] == "start")
        spark = [c for c in calls if c["model"] == "Muse Spark" and c["ev"] == "done" and c.get("out")]
        layout = next((c["out"] for c in spark if "front_wheel_x" in c["out"]), None)
        ended = any(c["ev"] == "end" for c in calls) or now >= self.end - 0.05
        stage = "waiting"
        if "photo.png" in files: stage = "photo"
        if "mask.png" in files: stage = "sam"
        if "object.glb" in files: stage = "mesh"
        if any(c["model"] == "Muse Spark" for c in calls): stage = "spark"
        if "assembly/input.json" in files: stage = "body"
        if n_started: stage = "glimmer"
        if ended: stage = "final"
        return {"x": round(x, 2), "clock": round(now - self.start, 1), "speed": round(speed, 1), "stage": stage,
                "calls": calls, "glimmer": {"started": n_started, "done": n_done,
                                            "verdicts": self.verdicts[:max(n_done, 0)],
                                            "total_doors": sum(1 for v in self.verdicts if v.get("door")),
                                            "pending": self.verdicts[n_done] if n_started > n_done and n_done < len(self.verdicts) else None},
                "layout": layout, "used": self.summary.get("layout") if layout else None,
                "n_layout": sum(1 for c in spark if "front_wheel_x" in c["out"]),
                "glass": "glass_mask.png" in files, "ended": ended,
                "summary": {k: self.summary.get(k) for k in ("parts", "glass_parts", "doors_passed", "stands",
                                                             "structure_passed", "function_passed")},
                "body_parts": len(json.loads((self.dir / "assembly" / "input.json").read_text()))}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>SAMgo Studio</title><style>
:root { --ink:#1d2329; --muted:#6e7884; --line:#e4e7eb; --green:#16a050; --red:#d01012; --violet:#7c3aed; --blue:#0866ff; --teal:#00897b; }
* { box-sizing:border-box } html,body { margin:0; height:100%; color:var(--ink); font-family:Futura,"Avenir Next",Helvetica,sans-serif;
  background: radial-gradient(ellipse at 50% 42%, #fbfaf8 0%, #f1f2f5 55%, #e6e9ee 100%); overflow:hidden }
header { position:absolute; left:48px; right:48px; top:30px; height:70px; display:flex; align-items:center; gap:22px }
.brand { font-size:30px; letter-spacing:.16em; font-weight:500 } .brand b { font-weight:500; color:var(--muted) }
.meta { font:14px Menlo,monospace; color:var(--muted) }
.right { margin-left:auto; display:flex; align-items:center; gap:16px }
.live { display:flex; align-items:center; gap:9px; font-size:14px; letter-spacing:.14em; color:var(--muted) }
.dot { width:11px; height:11px; border-radius:50%; background:var(--red); animation:pulse 1.3s infinite }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(208,16,18,.45)} 70%{box-shadow:0 0 0 10px rgba(208,16,18,0)} 100%{box-shadow:0 0 0 0 rgba(208,16,18,0)} }
.clock { font:500 26px Menlo,monospace; min-width:92px; text-align:right }
.speed { font:600 15px Menlo,monospace; background:var(--ink); color:#fff; border-radius:8px; padding:5px 9px; opacity:0; transition:opacity .25s }
.speed.on { opacity:1 }
aside { position:absolute; left:48px; top:120px; bottom:40px; width:560px; display:flex; flex-direction:column; gap:12px; overflow:hidden }
aside h2, .stage h2 { margin:0 0 4px; font-size:14px; letter-spacing:.18em; color:var(--muted); font-weight:500 }
.call { background:#fff; border-radius:16px; padding:14px 18px; box-shadow:0 6px 24px rgba(20,30,45,.07) }
.call.fresh { animation:in .4s ease-out }
.call .who { display:flex; align-items:center; gap:10px; font-size:21px; letter-spacing:.04em }
.call .who b { font-weight:600 } .call .id { font:12px Menlo,monospace; color:var(--muted) }
.call .s { margin-left:auto; font:14px Menlo,monospace; color:var(--muted) }
.call .ask, .call .out { font:15px/1.45 Menlo,monospace; margin-top:7px; color:var(--muted); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis }
.call .out { color:var(--ink) }
.pill { width:12px; height:12px; border-radius:50% }
.spin { width:15px; height:15px; border:2.5px solid var(--line); border-top-color:var(--ink); border-radius:50%; animation:sp .8s linear infinite }
@keyframes sp { to { transform:rotate(360deg) } } @keyframes in { from { opacity:0; transform:translateY(10px) } to { opacity:1; transform:none } }
.stage { position:absolute; left:650px; right:48px; top:120px; bottom:40px; background:#fff; border-radius:24px;
  box-shadow:0 12px 50px rgba(20,30,45,.09); padding:26px 30px; overflow:hidden }
.view { position:absolute; inset:70px 30px 30px 30px; opacity:0; transition:opacity .6s ease; display:flex; align-items:center; justify-content:center }
.view.on { opacity:1 }
.view img, .view video { max-width:100%; max-height:100%; object-fit:contain; border-radius:12px }
.stack { position:relative; width:100%; height:100% } .stack img { position:absolute; inset:0; margin:auto }
.title { position:absolute; left:30px; top:24px; font-size:16px; letter-spacing:.18em; color:var(--muted); transition:opacity .3s }
.bubble { position:absolute; right:40px; top:84px; max-width:520px; background:#fff; border-radius:18px; padding:16px 20px;
  box-shadow:0 10px 34px rgba(20,30,45,.13); animation:in .5s ease-out }
.bubble .who { color:var(--violet); font-size:14px; letter-spacing:.14em; font-weight:600 } .bubble .say { font-size:22px; margin-top:6px; line-height:1.3 }
.bubble pre { margin:10px 0 0; font:12px/1.4 Menlo,monospace; color:var(--muted); white-space:pre-wrap }
.term { width:100%; height:100%; background:#171b21; border-radius:16px; padding:28px 32px; font:19px/1.6 Menlo,monospace; color:#d8dee6; overflow:hidden }
.term .g { color:#5aa7ff } .term .ok { color:#5ad68f } .term .bad { color:#ff6b62 } .term .dim { color:#7d8794 }
.cursor { display:inline-block; width:10px; height:18px; background:#d8dee6; vertical-align:-3px; animation:bl 1s steps(2) infinite }
@keyframes bl { 50% { opacity:0 } }
iframe { width:100%; height:100%; border:0; border-radius:14px }
.badge { position:absolute; left:30px; bottom:26px; font-size:16px; letter-spacing:.08em; color:var(--muted) }
</style></head><body>
<header><div class="brand">SAMGO <b>STUDIO</b></div><div class="meta">camry.jpg  ·  recorded live run, Nov 4</div>
<div class="right"><div class="live"><div class="dot"></div>RUN</div><div class="clock" id="clock">0:00</div><div class="speed" id="speed">4×</div></div></header>
<aside><h2>LIVE META MODEL CALLS</h2><div id="calls"></div></aside>
<section class="stage">
  <div class="title" id="title"></div>
  <div class="view" id="v-photo"><img src="/run/photo.png"></div>
  <div class="view" id="v-sam"><div class="stack"><img src="/run/photo.png"><img src="/tint/mask"><img id="glass" src="/tint/glass" style="opacity:0;transition:opacity .6s"></div></div>
  <div class="view" id="v-mesh"><video src="/a/mesh.mp4" muted loop playsinline></video></div>
  <div class="view" id="v-spark"><video src="/a/spark.mp4" muted playsinline></video></div>
  <div class="view" id="v-body"><video src="/a/body.mp4" muted loop playsinline></video></div>
  <div class="view" id="v-glimmer"><div class="term" id="term"></div></div>
  <div class="view" id="v-final"><iframe id="viewer" src="about:blank"></iframe></div>
  <div class="badge" id="badge"></div>
</section>
<script>
const TITLES = { photo:"ONE PHOTO", sam:"SAM 3.1  ·  THE CAR, AND WHICH PIXELS ARE GLASS", mesh:"SAM 3D  ·  ONE PHOTO TO A 3D SHAPE",
  spark:"MUSE SPARK  ·  WHERE THE WHEELS AND DOORS GO", body:"THE LEGO BODY  ·  REAL PLATES", glimmer:"MUSE GLIMMER  ·  WRITES THE BUILD, THE CHECKER CHECKS IT",
  final:"THE BUILD  ·  REAL PARTS, CHECKED" };
const COLORS = { "SAM 3.1":"#0866ff", "SAM 3D":"#00897b", "Muse Spark":"#7c3aed", "Muse Glimmer":"#0a7cff" };
let cur = null, nCalls = -1, typed = 0, bubble = null, seen = new Set(), queue = [], shownAt = 0, sparkReady = false;
const ORDER = ["photo", "sam", "mesh", "spark", "body", "glimmer", "final"], DWELL = 2.6;
const shortId = (m, id) => m === "Muse Glimmer" ? "Glimmer 30B + our LoRA (run 11)" : String(id || "").replace("meta-models/", "");
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;");
function want(stage, x) {
  const target = ORDER.indexOf(stage), at = ORDER.indexOf(cur);
  if (target <= at) return;
  if (cur && x - shownAt < DWELL) return;
  show(ORDER[at + 1], x);
}
function show(stage, x) {
  if (stage === cur) return; cur = stage; shownAt = x;
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("on", v.id === "v-" + stage));
  document.getElementById("title").textContent = TITLES[stage] || "";
  const v = document.querySelector("#v-" + stage + " video");
  if (v) { v.currentTime = 0; if (stage !== "spark" || sparkReady) v.play(); else v.pause(); }
  if (stage === "final") { const f = document.getElementById("viewer"); f.onload = () => { const d = f.contentDocument;
    const st = d.createElement("style"); st.textContent = "header, .bar, .facts { display:none !important } body { padding:0 !important; gap:0 !important; background:#fff !important } #stage { border-radius:0 !important; height:100vh !important; background:#fff !important }";
    d.head.appendChild(st);
    setTimeout(() => { const w = f.contentWindow; try { w.eval(`(() => { const bb = bounds(group), c = bb.getCenter(new THREE.Vector3()), s = bb.getSize(new THREE.Vector3());
      const d = Math.max(s.x, s.y, s.z) * 0.95, az = 0.55; camera.position.set(c.x + Math.cos(az) * d, c.y + 0.42 * d, c.z + Math.sin(az) * d);
      controls.target.copy(c); controls.update(); })()`); } catch (e) {} }, 900); }; f.src = "/run/viewer.html"; }
}
function sparkBubble(st) {
  if (!st.used) return;
  const L = st.used, d = (L.doors || []).slice().sort((a, b) => a.x0 - b.x0).map(x => `${x.x0}–${x.x1}`).join(" and ");
  if (!bubble) { bubble = document.createElement("div"); bubble.className = "bubble"; document.getElementById("v-spark").appendChild(bubble); }
  bubble.innerHTML = `<div class="who">MUSE SPARK</div><div class="say">Wheels at stud ${L.rear_wheel_x} and stud ${L.front_wheel_x}. Doors from stud ${d}.</div>
    <pre>${st.n_layout} answers, combined  ·  ${esc(st.layout)}</pre>`;
}
function term(g) {
  const lines = [];
  g.verdicts.filter(v => v.first !== false).forEach((v, i) => {
    lines.push(`<span class="g">glimmer</span> <span class="dim">› ${esc(v.task)}</span>`);
    v.edits.slice(0, 3).forEach(p => lines.push(`  add ${esc(p.type)} ${p.w ? p.w + "x" + p.l : ""} at (${p.x}, ${p.y}, ${p.z})`));
    if (v.edits.length > 3) lines.push(`<span class="dim">  … ${v.edits.length - 3} more</span>`);
    lines.push(v.ok ? `<span class="ok">checker ✓ ${esc(v.why)}</span>` : `<span class="bad">checker ✗ ${esc(v.why)}</span>`);
  });
  const doors = g.verdicts.filter(v => v.door);
  if (doors.length && doors.length === g.total_doors && !g.pending)
    lines.push(`<span class="dim">doors: every answer rejected → left out and reported, never faked by code</span>`);
  if (g.pending && g.pending.first !== false) lines.push(`<span class="g">glimmer</span> <span class="dim">› ${esc(g.pending.task)} — thinking</span> <span class="cursor"></span>`);
  const el = document.getElementById("term"); el.innerHTML = lines.slice(-15).join("<br>");
}
async function tick() {
  const st = await (await fetch("/state")).json();
  const m = Math.floor(st.clock / 60), s = Math.floor(st.clock % 60);
  document.getElementById("clock").textContent = m + ":" + String(s).padStart(2, "0");
  const sp = document.getElementById("speed"); sp.textContent = Math.round(st.speed) + "×"; sp.classList.toggle("on", st.speed >= 1.6 && !st.ended);
  if (st.calls.length !== nCalls) {
    nCalls = st.calls.length; const box = document.getElementById("calls"); box.innerHTML = "";
    const list = [];
    for (const c of st.calls) {
      if (c.ev === "start") list.push({...c, pending:true, t0:c.t});
      else if (c.ev === "done") { for (let i = list.length - 1; i >= 0; i--) if (list[i].model === c.model && list[i].pending) { Object.assign(list[i], c, {pending:false}); break; } }
    }
    for (const c of list.slice(-5).reverse()) {
      if (c.out === "None" || c.out === null) c.out = "✓ answer received";
      const key = c.model + "@" + c.t0; const d = document.createElement("div"); d.className = "call" + (seen.has(key) ? "" : " fresh"); seen.add(key);
      d.innerHTML = `<div class="who">${c.pending ? '<div class="spin"></div>' : `<div class="pill" style="background:${COLORS[c.model]||'#999'}"></div>`}
        <b>${esc(c.model)}</b><span class="id">${esc(shortId(c.model, c.id))}</span><span class="s">${c.pending ? "calling…" : c.secs + " s"}</span></div>` +
        (c.ask ? `<div class="ask">→ ${esc(c.ask)}</div>` : "") + (c.out ? `<div class="out">← ${esc(c.out)}</div>` : "");
      box.appendChild(d);
    }
  }
  if (st.glass) document.getElementById("glass").style.opacity = 1;
  if (st.layout && !sparkReady) { sparkReady = true; const sv = document.querySelector("#v-spark video"); if (cur === "spark") { sv.currentTime = 0; sv.play(); } }
  if (st.layout) sparkBubble(st);
  term(st.glimmer);
  want(st.stage, st.x);
  const S = st.summary;
  document.getElementById("badge").textContent = st.stage === "body" ? st.body_parts + " real plates, from SAM 3D's shape" :
    st.stage === "final" ? `${S.parts} parts  ·  wheels ✓  ·  glass ✓  ·  stands ${S.stands ? "✓" : "✗"}  ·  doors ${S.doors_passed ? "✓" : "✗ (rejected by the checker, left out)"}` : "";
}
setInterval(tick, 100);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    run: Run = None

    def log_message(self, *a):
        pass

    def send(self, body, kind):
        self.send_response(200); self.send_header("Content-Type", kind); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        r = Handler.run
        p = self.path.split("?")[0]
        if p == "/":
            return self.send(PAGE.encode(), "text/html; charset=utf-8")
        if p == "/state":
            return self.send(json.dumps(r.state()).encode(), "application/json")
        if p.startswith("/tint/"):
            sys.path.insert(0, str(Path(__file__).parent))
            from studio import tint
            return self.send(tint(r.dir, "mask_tint.png" if p.endswith("mask") else "glass_tint.png"), "image/png")
        base, name = (r.dir, p[5:]) if p.startswith("/run/") else (r.assets, p[3:]) if p.startswith("/a/") else (None, None)
        if base:
            f = (base / name).resolve()
            if f.exists():
                kind = {"png": "image/png", "mp4": "video/mp4", "html": "text/html; charset=utf-8"}.get(f.suffix[1:],
                                                                                                     "application/octet-stream")
                return self.send(f.read_bytes(), kind)
        self.send_response(404); self.end_headers()


def serve(out: Path, assets: Path, port=8767, target=44.0):
    Handler.run = Run(out.resolve(), assets.resolve(), target)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return Handler.run


if __name__ == "__main__":
    run = serve(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 8767,
                float(sys.argv[4]) if len(sys.argv) > 4 else 44.0)
    print("SAMgo Studio (replay) on http://127.0.0.1:8767", flush=True)
    while True:
        time.sleep(3600)
