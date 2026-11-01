"""A TV on a turntable stand and a laptop with a hinged lid, sized from a photo's voxels, in real LEGO.

- TV: a stand plate on the table, a 2x2 turntable [3680 + 3679] at its centre, a column of bricks on the turntable
  top, and a screen wall (black bricks, one stud deep) on a base plate on the column. The screen swivels about the
  vertical axis.
- Laptop: a base (keyboard deck) of stacked plates; at its back edge a row of bar plates [60478] facing back, and
  one stud behind them clip plates [63868] facing forward under the lid, which stands open as a wall of bricks. The
  lid swings about a horizontal axis along the back edge (kinematics.clip_axis).

Every model is checked: valid parts, one connected structure, standing (joints held), and the motion clear.
"""

import numpy as np


def _plate(w, l, x, y, z, color, kind="plate"):
    return {"type": kind, "w": w, "l": l, "x": x, "y": y, "z": z, "color": color}


def _row(x0: int, n: int, y: int, z: int, l: int, color: str, kind: str = "brick", first: int = 0) -> list[dict]:
    """Bricks or plates of depth `l` along x covering x0..x0+n with the longest catalog lengths; `first`: the first
    piece's length (offsets the seams from a layer laid without it)."""
    out, x = [], x0
    while x < x0 + n:
        k = max(k for k in (8, 6, 4, 3, 2, 1) if x + k <= x0 + n and (x > x0 or not first or k <= first))
        out.append(_plate(k, l, x, y, z, color, kind))
        x += k
    return out


def _wall(x0: int, width: int, y: int, z0: int, layers: int, color: str) -> list[dict]:
    """A one-stud-deep wall of bricks, running bond (seams offset every other layer)."""
    out = []
    for n in range(layers):
        z = z0 + 3 * n
        if n % 2 == 0 or width < 3:
            out += _row(x0, width, y, z, 1, color)
        else:
            out += [_plate(1, 1, x0, y, z, color, "brick")] + _row(x0 + 1, width - 2, y, z, 1, color) + \
                [_plate(1, 1, x0 + width - 1, y, z, color, "brick")]
    return out


def _size(target, default):
    if target is None:
        return default
    occ = np.argwhere(np.asarray(target) > 0)
    if not len(occ):
        return default
    ext = occ.max(axis=0) - occ.min(axis=0) + 1
    return int(ext[0]), int(ext[1]), int(ext[2])


def build_tv(target: np.ndarray | None = None) -> dict:
    """The TV (screen width and height from the target's extent; width rounded to an even number of studs)."""
    w, _, h = _size(target, (16, 4, 30))
    w = max(8, min(24, w - w % 2))
    layers = max(3, min(8, (h - 8) // 3))
    c = w // 2 - 1
    parts = [_plate(8, 2, c - 3, 1, 0, "dark_gray"),                                      # stand
             {"type": "turntable_base", "w": 2, "l": 2, "x": c, "y": 1, "z": 1, "color": "black"},
             {"type": "turntable_top", "w": 2, "l": 2, "x": c, "y": 1, "z": 2, "color": "black"},
             _plate(2, 2, c, 1, 3, "black", "brick"),                                       # column
             _plate(2, 2, c, 1, 6, "black", "brick")]
    parts += _row(0, w, 1, 9, 2, "black", "plate")                                         # screen base, 2 deep
    parts += _wall(0, w, 1, 10, layers, "black")
    parts += _row(0, w, 1, 10 + 3 * layers, 1, "dark_gray", "plate")                       # top edge
    requirements = [{"name": "screen_swivels", "kind": "swing", "axis": "vertical", "degrees": 60, "count": 1}]
    screen = list(range(2, len(parts)))
    return {"parts": parts, "requirements": requirements, "moving": {"screen": (screen, 60)},
            "checks": checks(parts, requirements, {"screen": (screen, 60)}, "vertical")}


def build_laptop(target: np.ndarray | None = None) -> dict:
    """The laptop (width and depth from the target's extent; width even, depth at least 8: with less, the open
    lid tips it backwards - the stability check says so)."""
    w, d, _ = _size(target, (12, 8, 12))
    w = max(6, min(20, w - w % 2))
    d = max(8, min(12, d))
    yb = d - 3                                                       # bar plates cover y yb-1..yb+1 at z = 1
    deck = []
    def rows_to(end):  # 2-deep rows ending exactly at `end`, a 1-deep row at y = 0 if needed
        rows = [(y, 2) for y in range(end - 2, -1, -2)]
        return ([(0, end % 2)] if end % 2 else []) + rows[::-1]

    # Row boundaries of the two layers fall on opposite parities, so every seam of one layer is bridged by the other.
    for y, depth in rows_to(yb) + [(yb, 1)]:                         # bottom layer, up to and under the bars
        deck += _row(0, w, y, 0, depth, "light_gray", "plate")
    for y, depth in rows_to(yb - 1):                                 # top layer, up to the bars; x seams offset too
        deck += _row(0, w, y, 1, depth, "light_gray", "plate", first=2)
    bars = [{"type": "bar_plate", "w": 1, "l": 2, "x": x, "y": yb - 1, "z": 1, "facing": "+y", "color": "dark_gray"}
            for x in range(0, w)]
    clips = [{"type": "clip_plate", "w": 1, "l": 2, "x": x, "y": yb + 2, "z": 1, "facing": "-y", "color": "dark_gray"}
             for x in range(0, w)]
    lid = _row(0, w, yb + 2, 2, 2, "dark_gray", "plate") + _wall(0, w, yb + 3, 3, 3, "dark_gray")
    lid += _row(0, w, yb + 3, 12, 1, "dark_gray", "plate")
    parts = deck + bars + clips + lid
    requirements = [{"name": "lid_swings", "kind": "swing", "axis": "horizontal", "degrees": 30, "count": 1}]
    moving = {"lid": (list(range(len(deck) + len(bars), len(parts))), 30)}
    return {"parts": parts, "requirements": requirements, "moving": moving,
            "checks": checks(parts, requirements, moving, "horizontal")}


def checks(parts: list[dict], requirements: list[dict], moving: dict, axis: str) -> dict:
    from .kinematics import bound_motion, connected, locked, requirement_met
    from .mechanism_bases import PIVOT_AS, structural
    from .stability import stability
    from .technic import structural as technic_structural
    from .verifier import verify

    r = verify(technic_structural(structural(parts)), None)
    rule = [i for i in r.issues if i.code not in ("floating", "disconnected")]
    st = stability(locked(parts, PIVOT_AS))
    out = {"structure": (not rule, "; ".join(i.message for i in rule) or "valid parts, no overlaps"),
           "connected": (connected(parts), "one structure through studs and joints"),
           "stands": (st.passed, "; ".join(i.message for i in st.issues) or "stands")}
    for req in requirements:
        out[req["name"]] = requirement_met(parts, {k: v for k, v in req.items() if k != "name"})
    for name, (members, degrees) in moving.items():
        out[f"{name}_moves"] = bound_motion(parts, members, degrees, axis)
    return out
