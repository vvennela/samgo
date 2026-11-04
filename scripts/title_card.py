"""Title frames in the reference style: huge flat off-white capitals in Futura Medium, ~80% of the frame width, a little
letter-spacing, no shadow. Cuts on hard, then settles from 103% to 100% over 8 frames, then holds.

    python scripts/title_card.py W H out_dir [TEXT]   (then overlay the frames with ffmpeg at the word onset)
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = int(sys.argv[1]), int(sys.argv[2])
out = Path(sys.argv[3]); out.mkdir(exist_ok=True)
TEXT = sys.argv[4] if len(sys.argv) > 4 else "SAMGO"
FPS, SECONDS, SETTLE = 30, 3.2, 8
FONT = "/System/Library/Fonts/Supplemental/Futura.ttc"   # face 0 = Futura Medium
COLOR = (244, 242, 238)        # warm off-white, as in the reference
OPACITY = 0.94
TRACK = 0.06                   # extra space between letters, as a fraction of the font size
WIDTH = 0.80                   # share of the frame width the word spans


def word_width(font, size):
    return sum(font.getlength(c) for c in TEXT) + TRACK * size * (len(TEXT) - 1)


size = 100
while word_width(ImageFont.truetype(FONT, size + 4, index=0), size + 4) < WIDTH * W:
    size += 4


def frame(scale: float) -> Image.Image:
    s = int(size * scale)
    font = ImageFont.truetype(FONT, s, index=0)
    _, top, _, bottom = font.getbbox("SAMGO")          # cap height box
    x = (W - word_width(font, s)) / 2
    y = (H - (bottom - top)) / 2 - top + 0.04 * H     # a touch below centre, like the reference
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for c in TEXT:
        d.text((x, y), c, font=font, fill=COLOR + (int(255 * OPACITY),))
        x += font.getlength(c) + TRACK * s
    return img


img = None
for k in range(int(FPS * SECONDS)):
    if k <= SETTLE:
        u = k / SETTLE
        img = frame(1.03 - 0.03 * (1 - (1 - u) ** 3))
    img.save(out / f"t{k:03d}.png")
