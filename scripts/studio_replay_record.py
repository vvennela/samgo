"""Record SAMgo Studio replaying a logged live run (scripts/studio_run.py wrote <out>/calls.jsonl and the run's
artifacts): every call and artifact appears at its recorded time (x speed), then "Preview motion" in the viewer.

    python scripts/studio_replay_record.py out.mp4 <run out dir> [speed=1]
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from studio import serve_in_thread  # noqa: E402


def main():
    video, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    vids = out / "_video"
    vids.mkdir(exist_ok=True)
    for f in vids.glob("*.webm"):
        f.unlink()
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"])
        ctx = br.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(vids),
                             record_video_size={"width": 1920, "height": 1080})
        pg = ctx.new_page()
        serve_in_thread(out, 8766, replay_speed=speed)
        pg.goto("http://127.0.0.1:8766/")
        while True:
            st = json.loads(urllib.request.urlopen("http://127.0.0.1:8766/state").read())
            if any(c.get("ev") == "end" for c in st["calls"]):
                break
            pg.wait_for_timeout(400)
        pg.wait_for_timeout(2000)
        try:
            pg.wait_for_selector("#viewer", timeout=15000)
            pg.wait_for_timeout(3000)
            vf = pg.frame(url=lambda u: u.endswith("viewer.html"))
            vf.evaluate("() => { const b = document.getElementById('sim'); b.hidden = false; b.click(); }")
        except Exception as e:  # noqa: BLE001
            print("viewer:", e, flush=True)
        pg.wait_for_timeout(9000)
        ctx.close()
        br.close()
    webm = max(vids.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", str(video)], check=True)
    print("recorded", video, flush=True)


if __name__ == "__main__":
    main()
