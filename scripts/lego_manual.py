"""A LEGO-style building manual (PDF) from a build and its certified steps (samgo.manual): a cover, one panel per step
(the 3D model up to that step from the viewer, the new parts outlined, a callout box with the step's parts drawn and
counted, "turn the model over" where the build flips), and a parts inventory at the end.

    PYTHONPATH=. python scripts/lego_manual.py build.json manual.json parts.csv out.pdf "Title"
"""
import collections
import csv
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from playwright.sync_api import sync_playwright

from samgo.parts import COLORS
from samgo.viewer3d import EXTRA, facts_for, page

HIDE = ("header, .bar, .facts { display: none !important; } body { padding: 0 !important; gap: 0 !important; "
        "background: #ffffff !important; } #stage { border-radius: 0 !important; height: 100vh !important; }")
PAGE = (1240, 1754)          # A4 at 150 dpi
BG, CALLOUT, INK, NEW = (255, 255, 255), (214, 232, 247), (25, 25, 25), (255, 140, 0)
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_R = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(n, bold=True):
    try:
        return ImageFont.truetype(FONT if bold else FONT_R, n)
    except OSError:
        return ImageFont.load_default()


def rgb(name):
    if name in EXTRA:
        return (150, 200, 230)
    return COLORS.get(name, (0, (150, 150, 150)))[1]


def shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c)


NAMES = {"plate": "Plate", "brick": "Brick", "technic_brick": "Technic brick", "hinge_base": "Hinge plate 1x4 swivel base",
         "hinge_top": "Hinge plate 1x4 swivel top", "wheel": "Wheel", "axle": "Axle", "bush": "Bush"}
SPECIAL = {"hinge_base", "hinge_top", "wheel"}


def number(k):
    """The LEGO part number(s) (as in parts.csv)."""
    from samgo.parts import LDRAW_IDS
    t, a, b, _ = k
    return {"hinge_base": "2429", "hinge_top": "2430", "wheel": "rim 2994 + tyre 6578", "technic_brick": "3701",
            "axle": {11: "23948"}.get(a, f"axle {a}"), "bush": "3713"}.get(t) or LDRAW_IDS.get((t, a, b), "?")
COLOR_NAMES = {"red": "Red", "black": "Black", "dark_gray": "Dark Bluish Gray", "light_gray": "Light Bluish Gray",
               "trans_clear": "Trans-Clear", "white": "White", "blue": "Blue", "yellow": "Yellow"}


def key(p):
    t = p["type"]
    if t in ("wheel",):
        return (t, p.get("size", "30.4x14"), 0, "black")
    if t == "axle":
        return (t, p["length"], 0, p.get("color", "light_gray"))
    if t == "bush":
        return (t, 1, 0, p.get("color", "light_gray"))
    w, l = sorted((p.get("w", 1), p.get("l", 1)))
    return (t, w, l, p.get("color", "light_gray"))


def icon(k, s=26):
    """An isometric drawing of one part (studs on top): s = pixels per stud."""
    t, a, b, col = k
    c = rgb(col)
    if t == "wheel":
        im = Image.new("RGBA", (int(3.2 * s), int(3.2 * s)), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        r = int(1.5 * s)
        d.ellipse((4, 4, 4 + 2 * r, 4 + 2 * r), fill=(30, 30, 30), outline=INK, width=2)
        d.ellipse((4 + r * 0.45, 4 + r * 0.45, 4 + r * 1.55, 4 + r * 1.55), fill=(165, 170, 175), outline=INK, width=2)
        d.ellipse((4 + r * 0.9, 4 + r * 0.9, 4 + r * 1.1, 4 + r * 1.1), fill=(60, 60, 60))
        return im
    if t == "axle":
        L = int(a * s * 0.75)
        im = Image.new("RGBA", (L + 20, 40), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((4, 12, 4 + L, 26), radius=6, fill=(165, 170, 175), outline=INK, width=2)
        d.text((L / 2 - 4, 26), str(a), fill=INK, font=font(14))
        return im
    h = {"plate": 0.4, "hinge_base": 0.4, "hinge_top": 0.4, "brick": 1.2, "technic_brick": 1.2}.get(t, 0.4)
    w, l = a, b
    # isometric axes: x to the lower right, y to the upper right, z up
    ex, ey = (0.87 * s, 0.5 * s), (0.87 * s, -0.5 * s)
    H = h * s * 1.2
    W = int((w + l) * 0.87 * s + 20)
    Ht = int((w + l) * 0.5 * s + H + 30)
    im = Image.new("RGBA", (W, Ht), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ox, oy = 10, 10 + l * 0.5 * s + 6

    def P(x, y, z):
        return (ox + x * ex[0] + y * ey[0], oy + x * ex[1] + y * ey[1] - z * H)

    top = [P(0, 0, 1), P(w, 0, 1), P(w, l, 1), P(0, l, 1)]
    front = [P(0, 0, 0), P(w, 0, 0), P(w, 0, 1), P(0, 0, 1)]
    right = [P(w, 0, 0), P(w, l, 0), P(w, l, 1), P(w, 0, 1)]
    if col in EXTRA:
        c = (200, 225, 240)
    d.polygon(front, fill=shade(c, 0.78), outline=INK)
    d.polygon(right, fill=shade(c, 0.62), outline=INK)
    d.polygon(top, fill=c, outline=INK)
    if t == "technic_brick":  # holes on the long side
        for i in range(1, w):
            cx, cy = P(i, 0, 0.5)
            d.ellipse((cx - 0.22 * s, cy - 0.3 * s, cx + 0.22 * s, cy + 0.3 * s), fill=(20, 20, 20))
    if t in ("hinge_base", "hinge_top"):
        cx, cy = P(w / 2, 0, 0.5)
        d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=shade(c, 0.5), outline=INK)
    for i in range(w):
        for j in range(l):
            cx, cy = P(i + 0.5, j + 0.5, 1)
            rx, ry = 0.3 * s, 0.17 * s
            d.rectangle((cx - rx, cy - 0.18 * s, cx + rx, cy), fill=shade(c, 0.8))
            d.ellipse((cx - rx, cy - ry - 0.18 * s, cx + rx, cy + ry - 0.18 * s), fill=c, outline=INK)
            d.arc((cx - rx, cy - ry, cx + rx, cy + ry), 0, 180, fill=INK)
    return im


def callout(groups, maxw=1100):
    """The parts box: each part drawn with its count."""
    icons = [(icon(k), n) for k, n in groups]
    pad, gap = 22, 26
    rows, row, x = [], [], pad
    for im, n in icons:
        if row and x + im.width > maxw - pad:
            rows.append(row)
            row, x = [], pad
        row.append((im, n))
        x += im.width + gap
    if row:
        rows.append(row)
    rh = [max(im.height for im, _ in r) + 34 for r in rows]
    W = min(maxw, max(sum(im.width + gap for im, _ in r) for r in rows) + pad)
    box = Image.new("RGB", (W, sum(rh) + 2 * pad - 10), CALLOUT)
    d = ImageDraw.Draw(box)
    y = pad
    for r, h in zip(rows, rh):
        x = pad
        for im, n in r:
            box.paste(im, (x, y + (h - 34 - im.height)), im)
            d.text((x, y + h - 30), f"{n}x", fill=INK, font=font(24))
            x += im.width + gap
        y += h
    d.rectangle((0, 0, W - 1, box.height - 1), outline=(120, 150, 180), width=3)
    return box


def outline_new(before, after):
    """Outline the pixels that changed between the two renders (the new parts) in orange."""
    diff = ImageChops.difference(before.convert("RGB"), after.convert("RGB")).convert("L").point(lambda v: 255 if v > 18 else 0)
    diff = diff.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    ring = ImageChops.subtract(diff.filter(ImageFilter.MaxFilter(9)), diff)
    out = after.convert("RGB").copy()
    out.paste(Image.new("RGB", out.size, NEW), mask=ring)
    return out


def main():
    build, manual, parts_csv, out, title = sys.argv[1:6]
    parts = json.loads(Path(build).read_text())
    parts = parts["parts"] if isinstance(parts, dict) else parts
    man = json.loads(Path(manual).read_text())
    steps = man["steps"]
    import samgo.animation as anim  # the manual's certified order, not the video-game build order cars get
    anim.order = lambda ps, m=None: [next(k for k, st in enumerate(steps) if i in st["parts"]) for i in range(len(ps))]
    html = Path(tempfile.mkdtemp()) / "m.html"
    html.write_text(page([("model", parts, facts_for(parts), man)]))
    shots = Path(tempfile.mkdtemp())
    view = {"dist": 0.95, "az": 0.75, "elev": 0.55, "orbit": 1e9, "move": False, "fit": 0.45}
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"])
        pg = br.new_page(viewport={"width": 1100, "height": 760}, color_scheme="light")
        pg.goto(html.as_uri())
        pg.wait_for_function("typeof window.renderAt === 'function'", timeout=60000)
        pg.add_style_tag(content=HIDE)
        pg.wait_for_timeout(400)
        for k in range(len(steps)):
            flip = bool(steps[k].get("frame", {}).get("flipped"))
            for j, s in (("a", k - 1), ("b", k)):
                pg.evaluate("([o]) => window.renderAt(0, 0, o)", [dict(view, step=s, fitTo=k, flip=flip)])
                pg.locator("#stage canvas").screenshot(path=str(shots / f"{k:03d}{j}.png"))
        pg.evaluate("([o]) => window.renderAt(0, 0, o)", [dict(view, step=len(steps), az=0.45, elev=0.35, dist=1.1, fit=None)])
        pg.locator("#stage canvas").screenshot(path=str(shots / "cover.png"))
        br.close()

    pages = []
    # cover
    pgc = Image.new("RGB", PAGE, (196, 22, 28))
    d = ImageDraw.Draw(pgc)
    d.text((80, 90), title, fill=(255, 255, 255), font=font(84))
    d.text((80, 200), f"{len(parts)} parts  ·  {len(steps)} steps  ·  every step checked to hold", fill=(255, 230, 200), font=font(34, False))
    cov = Image.open(shots / "cover.png").convert("RGB")
    cov = cov.resize((PAGE[0] - 120, int(cov.height * (PAGE[0] - 120) / cov.width)))
    pgc.paste(cov, (60, 360))
    d.text((80, PAGE[1] - 120), "Built from one photo by SAMgo", fill=(255, 230, 200), font=font(30, False))
    pages.append(pgc)

    # steps: two per page
    panel_h = (PAGE[1] - 60) // 2
    cur = None
    for k, s in enumerate(steps):
        if k % 2 == 0:
            cur = Image.new("RGB", PAGE, BG)
            pages.append(cur)
        y0 = 30 + (k % 2) * panel_h
        d = ImageDraw.Draw(cur)
        groups = sorted(collections.Counter(key(parts[i]) for i in s["parts"]).items(), key=lambda kv: (kv[0][0], kv[0][3], kv[0][1], kv[0][2]))
        box = callout(groups, maxw=900)
        d.text((40, y0 + 6), str(k + 1), fill=INK, font=font(96))
        cur.paste(box, (220, y0 + 10))
        a = Image.open(shots / f"{k:03d}a.png")
        b = Image.open(shots / f"{k:03d}b.png")
        img = outline_new(a, b)
        room = panel_h - box.height - 50
        scale = min((PAGE[0] - 80) / img.width, room / img.height)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
        cur.paste(img, ((PAGE[0] - img.width) // 2, y0 + box.height + 30))
        flip_now = bool(s.get("frame", {}).get("flipped"))
        flip_before = k > 0 and bool(steps[k - 1].get("frame", {}).get("flipped"))
        if flip_now != flip_before and k > 0:
            d.text((60, y0 + panel_h - 70), "Turn the model over", fill=(196, 22, 28), font=font(40))
        if k % 2 == 0:
            d.line((40, 30 + panel_h - 4, PAGE[0] - 40, 30 + panel_h - 4), fill=(220, 220, 220), width=2)

    # inventory
    inv = collections.Counter(key(p) for p in parts)
    numbers = {}
    for r in csv.DictReader(open(parts_csv)):
        numbers.setdefault(r["color_name"], {})[r["part"]] = r["part"]
    pinv = Image.new("RGB", PAGE, BG)
    d = ImageDraw.Draw(pinv)
    d.text((60, 50), "Parts list", fill=INK, font=font(64))
    x, y, colw, rowh = 60, 160, 375, 0
    for k, n in sorted(inv.items(), key=lambda kv: (kv[0][3], kv[0][0], kv[0][1], kv[0][2])):
        im = icon(k, s=20)
        if x + colw > PAGE[0]:
            x, y = 60, y + rowh + 20
            rowh = 0
        if y + im.height + 70 > PAGE[1] - 40:
            pages.append(pinv)
            pinv = Image.new("RGB", PAGE, BG)
            d = ImageDraw.Draw(pinv)
            x, y, rowh = 60, 60, 0
        pinv.paste(im, (x, y), im)
        t, a, b, col = k
        label = f"{n}x  {NAMES.get(t, t)} " + (f"{a}x{b}" if t in ("plate", "brick", "technic_brick") else "" if t in SPECIAL else str(a))
        label += f"  ({number(k)})"
        d.text((x, y + im.height + 4), label, fill=INK, font=font(18))
        d.text((x, y + im.height + 26), COLOR_NAMES.get(col, col), fill=(90, 90, 90), font=font(16, False))
        rowh = max(rowh, im.height + 50)
        x += colw
    pages.append(pinv)
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=150)
    print(out, len(pages), "pages", flush=True)


if __name__ == "__main__":
    main()
