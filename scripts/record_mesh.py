"""Record a turning clip of a SAM 3D mesh (.glb) with its own colours (three.js in headless Chromium -> mp4).
    python scripts/record_mesh.py object.glb out.mp4 seconds"""
import base64
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

FPS, W, H = 30, 1920, 1080
PAGE = """<html><body style="margin:0;background:#101316">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script>
const r = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true}); r.setSize(%d, %d);
r.outputEncoding = THREE.sRGBEncoding; document.body.appendChild(r.domElement);
const s = new THREE.Scene(); s.background = new THREE.Color("#101316");
s.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.1)); const d = new THREE.DirectionalLight(0xffffff, 0.8);
d.position.set(3, 5, 4); s.add(d);
const cam = new THREE.PerspectiveCamera(35, %d / %d, 0.01, 100);
const bin = Uint8Array.from(atob("%s"), c => c.charCodeAt(0)).buffer;
let obj = null, c = new THREE.Vector3(), R = 1;
new THREE.GLTFLoader().parse(bin, "", g => { obj = g.scene; s.add(obj);
  const bb = new THREE.Box3().setFromObject(obj); bb.getCenter(c); R = bb.getSize(new THREE.Vector3()).length() * 1.15; window.READY = true; });
window.at = t => { const a = 0.6 + 2 * Math.PI * t / 12; cam.position.set(c.x + Math.cos(a) * R, c.y + R * 0.35, c.z + Math.sin(a) * R);
  cam.lookAt(c); r.render(s, cam); };
</script></body></html>"""


def main():
    glb, out, secs = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    html = Path(tempfile.mkdtemp()) / "mesh.html"
    html.write_text(PAGE % (W, H, W, H, base64.b64encode(glb.read_bytes()).decode()))
    frames = Path(tempfile.mkdtemp())
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"])
        pg = br.new_page(viewport={"width": W, "height": H})
        pg.goto(html.as_uri())
        pg.wait_for_function("window.READY === true", timeout=120000)
        for k in range(int(secs * FPS)):
            pg.evaluate("t => window.at(t)", k / FPS)
            pg.locator("canvas").screenshot(path=str(frames / f"{k:05d}.png"))
        br.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    print(out)


if __name__ == "__main__":
    main()
