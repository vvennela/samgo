"""Record the SAMgo Studio presentation view (scripts/studio_show.py) replaying a logged live run, then press
"Preview motion" in the finished model's viewer.

    python scripts/studio_show_record.py out.mp4 <run out dir> <assets dir> [target seconds=44]
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from studio_show import serve  # noqa: E402


def main():
    video, out, assets = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
    target = float(sys.argv[4]) if len(sys.argv) > 4 else 44.0
    vids = Path(out).resolve() / "_video_show"
    vids.mkdir(exist_ok=True)
    for f in vids.glob("*.webm"):
        f.unlink()
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist",
                                      "--autoplay-policy=no-user-gesture-required"])
        ctx = br.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(vids),
                             record_video_size={"width": 1920, "height": 1080})
        pg = ctx.new_page()
        serve(out, assets, 8767, target)
        pg.goto("http://127.0.0.1:8767/")
        while True:
            st = json.loads(urllib.request.urlopen("http://127.0.0.1:8767/state").read())
            if st["ended"]:
                break
            pg.wait_for_timeout(300)
        pg.wait_for_timeout(3500)
        try:
            vf = pg.frame(url=lambda u: u.endswith("viewer.html"))
            vf.evaluate("() => { const b = document.getElementById('sim'); b.hidden = false; b.click(); }")
        except Exception as e:  # noqa: BLE001
            print("viewer:", e, flush=True)
        pg.wait_for_timeout(8000)
        ctx.close()
        br.close()
    webm = max(vids.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1.6", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", str(video)], check=True)
    print("recorded", video, flush=True)


if __name__ == "__main__":
    main()
