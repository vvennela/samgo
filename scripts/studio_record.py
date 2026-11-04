"""Record a live SAMgo run in SAMgo Studio: a browser at 1920x1080 records the Studio page while the real pipeline runs
(scripts/studio_run.py, every live model call logged), then presses "Preview motion" in the finished model's viewer.

    PYTHONPATH=.:<fireworks cookbook> python scripts/studio_record.py out.mp4 <out dir> <pipeline args...>
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from studio import serve_in_thread  # noqa: E402


def main():
    video, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    args = sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "calls.jsonl"):
        (out / f).unlink(missing_ok=True)
    serve_in_thread(out, 8765)
    vids = out / "_video"
    vids.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"])
        ctx = br.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(vids),
                             record_video_size={"width": 1920, "height": 1080})
        pg = ctx.new_page()
        pg.goto("http://127.0.0.1:8765/")
        pg.wait_for_timeout(1500)
        t0 = time.time()
        log = open(out / "pipeline.log", "w")
        proc = subprocess.Popen([sys.executable, str(HERE / "studio_run.py"), *args, "-o", str(out)],
                                stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        while proc.poll() is None:
            pg.wait_for_timeout(500)
        print(f"pipeline exit {proc.returncode} after {time.time() - t0:.0f}s", flush=True)
        pg.wait_for_timeout(2500)                               # the final card appears
        try:
            pg.wait_for_selector("#viewer", timeout=15000)
            pg.wait_for_timeout(2500)                            # the viewer loads the model
            vf = pg.frame(url=lambda u: u.endswith("viewer.html"))
            vf.evaluate("() => { const b = document.getElementById('sim'); b.hidden = false; b.click(); }")
        except Exception as e:  # noqa: BLE001
            print("no motion button:", e, flush=True)
        pg.wait_for_timeout(9000)
        ctx.close()
        br.close()
    webm = max(vids.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", str(video)], check=True)
    s = json.loads((out / "summary.json").read_text())
    print("recorded", video, "status", s.get("status"), "parts", s.get("parts"), flush=True)


if __name__ == "__main__":
    main()
