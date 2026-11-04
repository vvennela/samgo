"""The manual's finale, in one continuous shot on the last page: the step page as printed, then its hovering parts
slide home as the arrows fade, the step number and parts box fade away, and the finished model does the LEGO-game
bounce on the page. Rendered with the manual's own camera and look (scripts/lego_manual.py), placed exactly where the
manual put the step image, so the first frame is the printed page.

    PYTHONPATH=. python scripts/film_finale.py build.json manual.json last_page.png out_dir
writes out_dir/%05d.png (whole pages, 1754x1240)
"""
import collections
import json
import math
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lego_manual as lm  # noqa: E402
from samgo.viewer3d import facts_for, page  # noqa: E402

FPS = 30
LIFT = 48
HOLD, SLIDE, CHROME_OUT, BOUNCE_AT, END = 0.35, (0.4, 1.0), (1.0, 1.45), 1.5, 3.7
KEYS = [(0.00, 0, 1.00, 1.00), (0.12, 0, 1.05, 0.92), (0.28, 70, 0.97, 1.06), (0.44, 95, 1.00, 1.00),
        (0.62, 0, 1.06, 0.91), (0.74, 14, 0.99, 1.03), (0.86, 0, 1.02, 0.98), (0.96, 0, 1.00, 1.00)]


def bounce(t):
    for (t0, *a), (t1, *b) in zip(KEYS, KEYS[1:]):
        if t0 <= t <= t1:
            u = 0.5 - 0.5 * math.cos(math.pi * (t - t0) / (t1 - t0))
            return [x + (y - x) * u for x, y in zip(a, b)]
    return KEYS[-1][1:]


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def main():
    build, manual, last_page, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    parts = json.loads(Path(build).read_text())
    parts = parts["parts"] if isinstance(parts, dict) else parts
    man = json.loads(Path(manual).read_text())
    steps = man["steps"]
    k = len(steps) - 1
    import samgo.animation as anim  # the manual's own order, as lego_manual sets it
    anim.order = lambda ps, m=None: [next(j for j, st in enumerate(steps) if i in st["parts"]) for i in range(len(ps))]
    html = Path(tempfile.mkdtemp()) / "m.html"
    html.write_text(page([("model", parts, facts_for(parts), man)]))
    view = {"dist": 0.9, "az": 0.75, "elev": 0.6, "orbit": 1e9, "move": False, "fit": 0.4, "edges": True}
    shots = {}
    lifts = [LIFT * (1 - ease(i / 17)) for i in range(18)]
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"])
        pg = br.new_page(viewport={"width": 1500, "height": 960}, color_scheme="light")
        pg.goto(html.as_uri())
        pg.wait_for_function("typeof window.renderAt === 'function'", timeout=60000)
        pg.add_style_tag(content=lm.HIDE)
        pg.wait_for_timeout(400)
        shot = lambda: Image.open(__import__("io").BytesIO(pg.locator("#stage canvas").screenshot())).convert("RGBA")
        marks = pg.evaluate("([o]) => window.renderAt(0, 0, o)", [dict(view, step=k, fitTo=k, lift=LIFT)])
        shots["printed"] = shot().convert("RGB")
        lm.arrows(shots["printed"], marks if isinstance(marks, list) else [])
        for i, L in enumerate(lifts):
            pg.evaluate("([o]) => window.renderAt(0, 0, o)", [dict(view, step=k, fitTo=k, lift=L)])
            shots[f"lift{i}"] = shot().convert("RGB")
        # the finished model alone (transparent, no floor) and the floor alone, for the bounce
        pg.evaluate("([o]) => window.renderAt(0, 0, o)", [dict(view, step=k, fitTo=k, lift=0)])
        pg.evaluate("""() => { renderer.setClearColor(0x000000, 0); scene.background = null;
            scene.traverse(o => { if (o.isMesh && o.geometry && o.geometry.type === 'PlaneGeometry') o.visible = false; });
            renderer.render(scene, camera); }""")
        pg.add_style_tag(content="html, body, #stage { background: transparent !important; }")
        shots["car"] = Image.open(__import__("io").BytesIO(
            pg.locator("#stage canvas").screenshot(omit_background=True))).convert("RGBA")
        pg.evaluate("""() => { scene.traverse(o => { if (o.isMesh && o.geometry && o.geometry.type === 'PlaneGeometry') o.visible = true;
            else if ((o.isMesh || o.isInstancedMesh || o.isLineSegments) && o.parent) o.visible = false; });
            bg(); renderer.render(scene, camera); }""")
        shots["floor"] = shot().convert("RGB")
        br.close()
    import os
    if os.environ.get("FINALE_DEBUG"):
        for k2 in ("floor", "car", "lift17"):
            shots[k2].save(out / f"_debug_{k2}.png")

    # the manual's layout for this step: the crop box of the printed image, its scale and place on the page
    LAND = lm.LAND
    groups = sorted(collections.Counter(lm.key(parts[i]) for i in steps[k]["parts"]).items(),
                    key=lambda kv: (kv[0][0], kv[0][3], kv[0][1], kv[0][2]))
    box = lm.callout(groups, maxw=1000)
    printed = shots["printed"]
    bgc = printed.getpixel((5, 5))
    diff = ImageChops.difference(printed, Image.new("RGB", printed.size, bgc)).convert("L").point(lambda v: 255 if v > 40 else 0)
    b = diff.getbbox()
    cb = (max(0, b[0] - 50), max(0, b[1] - 50), min(printed.width, b[2] + 50), min(printed.height, b[3] + 50))
    y_img = 40 + box.height + 24
    cw, ch = cb[2] - cb[0], cb[3] - cb[1]
    scale = min((LAND[0] - 80) / cw, (LAND[1] - y_img - 30) / ch)
    W, H = int(cw * scale), int(ch * scale)
    px, py = (LAND[0] - W) // 2, y_img
    place = lambda im: im.crop(cb).resize((W, H), Image.LANCZOS)

    page_img = Image.open(last_page).convert("RGB").resize(LAND)
    plain = Image.new("RGB", LAND, page_img.getpixel((5, 5)))
    chrome_mask = Image.new("L", LAND, 255)
    ImageDraw.Draw(chrome_mask).rectangle((px, py, px + W, py + H), fill=0)

    car = place(shots["car"])
    floor = place(shots["floor"])
    cbox = car.getchannel("A").point(lambda v: 255 if v > 30 else 0).getbbox()
    car_bottom = cbox[3]
    n = int(END * FPS)
    for f in range(n):
        t = f / FPS
        frame = plain.copy()
        if t < SLIDE[0]:
            area = place(printed) if t < HOLD else Image.blend(place(printed), place(shots["lift0"]),
                                                                 ease((t - HOLD) / (SLIDE[0] - HOLD)))
        elif t < BOUNCE_AT:
            i = min(len(lifts) - 1, int(round((len(lifts) - 1) * ease((t - SLIDE[0]) / (SLIDE[1] - SLIDE[0])))))
            area = place(shots[f"lift{i}"])
        else:
            lift, sx, sy = bounce(t - BOUNCE_AT)
            area = floor.copy().convert("RGBA")
            # a soft contact shadow under the car, shrinking while it is up
            sh = Image.new("L", area.size, 0)
            cx = (cbox[0] + cbox[2]) / 2
            r = (cbox[2] - cbox[0]) * 0.38 * (1 - 0.25 * min(1, lift / 95))
            ImageDraw.Draw(sh).ellipse((cx - r, car_bottom - 30, cx + r, car_bottom + 18), fill=int(70 * (1 - 0.5 * min(1, lift / 95))))
            sh = sh.filter(ImageFilter.GaussianBlur(16))
            area.paste(Image.new("RGBA", area.size, (30, 40, 55, 255)), mask=sh)
            cw2, ch2 = int(car.width * sx), int(car.height * sy)
            c2 = car.resize((cw2, ch2), Image.LANCZOS)
            ox = int(cx - cx * sx)
            oy = int(car_bottom - car_bottom * sy - lift * scale)
            area.alpha_composite(c2, (max(-cw2, ox), oy) if oy >= 0 else (ox, 0))
            area = area.convert("RGB")
        frame.paste(area, (px, py))
        a = 1 - ease((t - CHROME_OUT[0]) / (CHROME_OUT[1] - CHROME_OUT[0]))
        if a > 0:
            frame.paste(page_img, (0, 0), chrome_mask.point(lambda v: int(v * a)))
        frame.save(out / f"{f:05d}.png")
    print(n, "frames ->", out)


if __name__ == "__main__":
    main()
