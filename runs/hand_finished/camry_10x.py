"""Hand-finished 10x Camry (200 studs long, ~1.6 m): the resolution sweep's body (SAM 3D mesh voxelized at 200 studs)
painted, glazed and re-laid as plates outside / bricks inside.

- Paint: the photo's grays become the body colour; black (the tyres) stays black.
- Glass: SAM's glass is a blotch at this size, so the windows come from the shape: every outer cell of the cabin
  (above the window line ZB, below the flat roof's last ROOF plates, where the cabin rises at least 8 plates above the
  line) is trans-clear, except A/C pillars (cabin corners) and a B pillar (middle of each side). Cells just behind the
  glass are black, so the windows read dark.
- Layout: one cell of plates on the outside; bricks wherever all three of their layers are inside.

    python runs/hand_finished/camry_10x.py grid.npz out.json [ZB] [paint]
"""
import json
import sys
import time

import numpy as np

from samgo.builder import _layout
from samgo.parts import COLOR_INDEX

ROOF, P, BEHIND = 5, 4, 3


def neighbours_air(occ: np.ndarray) -> np.ndarray:
    pad = np.pad(occ, 1, constant_values=False)
    air = np.zeros_like(occ)
    for ax in range(3):
        for d in (1, -1):
            air |= ~np.roll(pad, d, axis=ax)[1:-1, 1:-1, 1:-1]
    return air & occ


def dilate(m: np.ndarray, k: int) -> np.ndarray:
    out = m.copy()
    for _ in range(k):
        pad = np.pad(out, 1, constant_values=False)
        grown = out.copy()
        for ax in range(3):
            for d in (1, -1):
                grown |= np.roll(pad, d, axis=ax)[1:-1, 1:-1, 1:-1]
        out = grown
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ZB = int(sys.argv[3]) if len(sys.argv) > 3 else 118
    paint = sys.argv[4] if len(sys.argv) > 4 else "red"
    g = np.load(src)["grid"]
    occ = g > 0
    X, Y, Z = g.shape
    shell = neighbours_air(occ)
    zz = np.arange(Z)[None, None, :]
    ztop = np.array([np.nonzero(occ[x].any(axis=0))[0].max() if occ[x].any() else -1 for x in range(X)])
    peak = int(np.argmax(ztop))
    # the cabin: where the profile rises well above the window line (behind the roof, higher: the trunk lid sits
    # just above the line and is not glass)
    cabin_x = np.where(np.arange(X) < peak, ztop >= ZB + 8, ztop >= ZB + 14)
    ZR = int(ztop.max()) - ROOF  # the flat roof: its last ROOF plates stay paint (a sloped windshield is the top)
    band = occ & (zz >= ZB) & (zz <= ZR) & cabin_x[:, None, None]
    cx = np.nonzero(cabin_x)[0]
    xm = (cx.min() + cx.max()) // 2  # the B pillar: one position, the middle of the cabin
    glass = shell & band
    # pillars, per layer: the cabin's corners (A front, C back) and the middle of each side (B)
    for z in range(ZB, Z):
        lay = band[:, :, z]
        if not lay.any():
            continue
        xs, ys = np.nonzero(lay)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        side = np.zeros(Y, bool)
        side[:y0 + P + 1] = True
        side[y1 - P:] = True
        cols = np.zeros(X, bool)
        cols[x0:x0 + P + 1] = True
        cols[x1 - P:x1 + 1] = True
        cols[xm - 2:xm + 3] = True
        glass[:, :, z] &= ~(cols[:, None] & side[None, :])
    from scipy import ndimage

    lab, n = ndimage.label(glass, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())
    glass &= sizes[lab] >= 200  # small patches (a mirror at window height) are not windows
    behind = dilate(glass, BEHIND) & occ & ~shell
    colour = np.where(g == COLOR_INDEX["black"], COLOR_INDEX["black"], COLOR_INDEX[paint]) * occ
    colour[behind] = COLOR_INDEX["black"]
    PLACE = COLOR_INDEX["tan"]  # placeholder the builder knows for glass (the car has no tan)
    colour[glass] = PLACE
    inside = occ & ~shell
    print(f"cells {int(occ.sum())}, shell {int(shell.sum())}, glass {int(glass.sum())}, dark behind glass "
          f"{int(behind.sum())}, inside {int(inside.sum())}", flush=True)
    if "--preview" in sys.argv:  # projections from the side, front, back and top instead of the layout
        from PIL import Image

        pal = np.zeros((max(COLOR_INDEX.values()) + 1, 3), np.uint8)
        from samgo.parts import COLORS
        for n, i in COLOR_INDEX.items():
            pal[i] = COLORS[n][1]
        pal[0], pal[PLACE] = (255, 255, 255), (70, 110, 150)
        for name, axis, flip in (("side", 1, False), ("front", 0, False), ("back", 0, True), ("top", 2, True)):
            o = np.moveaxis(colour, axis, 0)
            o = o[::-1] if flip else o
            im = pal[np.take_along_axis(o, np.argmax(o > 0, axis=0)[None], 0)[0]].transpose(1, 0, 2)
            im = im if name == "top" else im[::-1]
            Image.fromarray(im).resize((im.shape[1] * 3, im.shape[0] * 3), Image.NEAREST).save(f"{dst}_{name}.png")
        return
    t = time.time()
    parts = _layout(colour.astype(np.int32), None, bricks=True, brick_ok=inside)
    for q in parts:
        if q["color"] == "tan":
            q["color"] = "trans_clear"
    print(f"layout {time.time() - t:.0f}s: {len(parts)} parts, bricks {sum(q['type'] == 'brick' for q in parts)}, "
          f"glass {sum(q['color'] == 'trans_clear' for q in parts)}", flush=True)
    json.dump(parts, open(dst, "w"))
    print("saved", dst, flush=True)


if __name__ == "__main__":
    main()
