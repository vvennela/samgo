"""The drawing, marked up: Muse Image's LEGO-looking car with what no real LEGO build could have circled one by one,
then the line under it. Frames for the film (1920x1080, 30 fps).

    python scripts/film_drawing.py drawing.png bg.png out_dir
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FUT = "/System/Library/Fonts/Supplemental/Futura.ttc"
RED = (208, 16, 18)
INK = (29, 35, 41)
FPS, SECONDS = 30, 5.4
# (box in the drawing's own pixels 1920x1280, label, label side, appears at s)
MARKS = [((190, 520, 660, 720), "BENT PLATES: NO SUCH PART", "above", 0.6),
         ((1270, 455, 1400, 565), "WHAT HOLDS THE MIRROR?", "above", 1.4),
         ((1440, 500, 1525, 840), "DOORS WITH NO HINGES", "left", 2.2),
         ((1590, 430, 1790, 790), "BRICKS THAT CURVE: NO SUCH PART", "below", 3.0)]
CAPTION_AT = 3.8


def main():
    drawing, bg, out = Image.open(sys.argv[1]).convert("RGBA"), Image.open(sys.argv[2]).convert("RGBA"), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    W = 1400
    k = W / drawing.width
    d_img = drawing.resize((W, int(drawing.height * k)), Image.LANCZOS)
    ox, oy = (1920 - W) // 2, 24
    f_lab = ImageFont.truetype(FUT, 30, index=0)
    f_cap = ImageFont.truetype(FUT, 44, index=0)
    cap = "A PICTURE: NO PARTS LIST, NOTHING TO CHECK"
    for n in range(int(SECONDS * FPS)):
        t = n / FPS
        im = bg.copy()
        im.alpha_composite(d_img, (ox, oy))
        dr = ImageDraw.Draw(im)
        for (x0, y0, x1, y1), label, side, at in MARKS:
            u = (t - at) / 0.3
            if u <= 0:
                continue
            a = int(255 * min(1, u))
            grow = 1 + 0.25 * max(0.0, 1 - u)
            cx, cy = ox + (x0 + x1) / 2 * k, oy + (y0 + y1) / 2 * k
            hw, hh = (x1 - x0) / 2 * k * grow + 8, (y1 - y0) / 2 * k * grow + 8
            dr.rounded_rectangle([cx - hw, cy - hh, cx + hw, cy + hh], 14, outline=RED + (a,), width=6)
            tw = f_lab.getlength(label)
            if side == "above":
                lx, ly = cx - tw / 2, cy - hh - 46
                if label.startswith("BRICKS"):
                    lx, ly = cx + hw - tw, cy - hh - 46
            elif side == "below":
                lx, ly = cx - tw / 2, cy + hh + 10
                if label.startswith("BRICKS"):
                    lx = cx + hw - tw
            elif side == "left":
                lx, ly = cx - hw - tw - 18, cy - 18
            else:
                lx, ly = min(1920 - tw - 30, cx + hw + 14), cy - 18
                if lx + tw > 1900:
                    lx, ly = cx - tw / 2, cy + hh + 10
            lx = max(20, min(1900 - tw, lx))
            pad = 10
            dr.rounded_rectangle([lx - pad, ly - 4, lx + tw + pad, ly + 40], 10, fill=(255, 255, 255, int(a * 0.92)))
            dr.text((lx, ly), label, font=f_lab, fill=RED + (a,))
        u = (t - CAPTION_AT) / 0.35
        if u > 0:
            a = int(255 * min(1, u))
            tw = f_cap.getlength(cap)
            dr.text(((1920 - tw) / 2, 1000), cap, font=f_cap, fill=INK + (a,))
        im.convert("RGB").save(out / f"{n:05d}.png")
    print(int(SECONDS * FPS), "frames ->", out)


if __name__ == "__main__":
    main()
