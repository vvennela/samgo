"""Reference mechanisms for non-car objects: code places one correct mechanism, so training rows have an answer.

Each placer takes a voxel target, puts real pivoting parts where the object's own shape says the motion is, and
builds the rest with the S4 builder around them. The generic motion checker (kinematics.requirement_met) then
decides whether it works; only builds that pass are kept as bases.

- turntable (TV, monitor): a 2x2 turntable (3680 base + 3679 top) at the stand's narrowest layer; the screen
  turns about a vertical axis.
"""

import numpy as np

from .builder import build
from .kinematics import connected, locked, requirement_met
from .verifier import MatchConfig, verify
from .parts import COLOR_INDEX
from .stability import stability

PIVOT_AS = {"turntable_base": "plate", "turntable_top": "plate", "hinge_base": "plate", "hinge_top": "plate",
            "click_a": "plate", "click_b": "plate", "car_door": "brick"}
REQUIREMENTS = {
    "turntable": {"kind": "swing", "axis": "vertical", "degrees": 60},
    "lid": {"kind": "swing", "axis": "horizontal", "degrees": 60},
    "door": {"kind": "swing", "axis": "vertical", "degrees": 90},
}


def structural(parts: list[dict]) -> list[dict]:
    """What the S4 builder and S3a verifier see: pivoting parts as the plain plates/bricks they occupy."""
    return [p | {"type": PIVOT_AS[p["type"]]} if p["type"] in PIVOT_AS else p for p in parts]


def _keep_connected(parts: list[dict]) -> list[dict]:
    """Drop stray parts that neither rest on the structure nor hang on a joint (cutting a joint gap can leave
    some); the pivots come first and are always kept."""
    from .kinematics import analyse

    bodies, joints = analyse(parts)
    root = {i: r for r, members in bodies.items() for i in members}
    linked = {r for a, b, *_ in joints for r in (a, b)}
    big = max(bodies, key=lambda r: len(bodies[r]))
    keep = {big} | linked
    return [p for i, p in enumerate(parts) if root[i] in keep]


def _verdict(parts: list[dict], target: np.ndarray, mechanism: str) -> dict:
    """Structure (S3a rules, with joints counting as links), the motion requirement, and stability with the
    joints held in their current pose."""
    report = verify(structural(parts), target, config=MatchConfig())
    rule_issues = [i.code for i in report.issues if i.code not in ("floating", "disconnected")]
    structure = not rule_issues and connected(parts)
    ok, why = requirement_met(parts, REQUIREMENTS[mechanism])
    stands = stability(locked(parts, PIVOT_AS)).passed
    return {"parts": parts, "target": target, "structure": structure, "moves": ok, "why": why, "stands": stands,
            "mechanism": mechanism, "issues": rule_issues}


def _neck(occ: np.ndarray) -> tuple[int, int, int] | None:
    """(layer z, center x, center y) of the narrowest horizontal layer between a base and something wider above."""
    X, Y, Z = occ.shape
    area = occ.sum(axis=(0, 1))
    zs = [z for z in range(2, Z - 4) if area[z] > 0]
    if not zs:
        return None
    z = min(zs, key=lambda k: (area[k], -k))
    if area[z] * 2 > max(area[z + 1:].max(initial=0), 1) or area[:z].max(initial=0) == 0:
        return None  # no clear neck: nothing much wider above, or nothing below
    xs, ys = np.nonzero(occ[:, :, z])
    return z, int(round(xs.mean())), int(round(ys.mean()))


def turntable(target: np.ndarray, attempts: int = 4) -> dict | None:
    """A TV/monitor with its screen on a turntable at the neck of its stand; None if no neck or it fails."""
    occ = target > 0
    neck = _neck(occ)
    if neck is None:
        return None
    z, cx, cy = neck
    X, Y, Z = target.shape
    x0, y0 = min(max(cx - 1, 0), X - 2), min(max(cy - 1, 0), Y - 2)
    shaped = target.copy()
    color = COLOR_INDEX["black"]
    shaped[:, :, z:z + 2] = 0                       # the two plates of the turntable are the only link
    shaped[x0:x0 + 2, y0:y0 + 2, z:z + 2] = color
    # The parts touching the turntable from below and above must exist so it is held on both sides.
    if not occ[x0:x0 + 2, y0:y0 + 2, z - 1].any() or not occ[x0:x0 + 2, y0:y0 + 2, z + 2].any():
        return None
    pivots = [{"type": "turntable_base", "w": 2, "l": 2, "x": x0, "y": y0, "z": z, "color": "black"},
              {"type": "turntable_top", "w": 2, "l": 2, "x": x0, "y": y0, "z": z + 1, "color": "black"}]
    body = build(shaped, attempts=attempts, given=structural(pivots), anchored=True)
    parts = _keep_connected(pivots + body.pieces[len(pivots):])
    return _verdict(parts, shaped, "turntable")


def lid(target: np.ndarray, attempts: int = 4) -> dict | None:
    """A laptop: the screen hinged to the base with click hinges (44301/44302) along the line where it rises.

    The long horizontal axis whose far end is much taller than its near end is the depth (front -> back); the
    base's thickness is the median height of its front third; the hinge row is the first row from the front
    where the shape rises well above the base. The screen keeps only the hinge as its link to the base.
    """
    occ = target > 0
    if not occ.any():
        return None
    height = np.where(occ.any(axis=2), occ.shape[2] - np.argmax(occ[:, :, ::-1], axis=2), 0)
    best = None
    for axis in (0, 1):
        h = height if axis == 1 else height.T          # rows along `axis` index 1 after this
        n = h.shape[1]
        front, back = h[:, : max(n // 3, 1)], h[:, -max(n // 3, 1):]
        lift = back[back > 0].mean() - front[front > 0].mean() if (front > 0).any() and (back > 0).any() else 0
        for flip in (False, True):
            hh = h[:, ::-1] if flip else h
            f = hh[:, : max(n // 3, 1)]
            t = int(np.median(f[f > 0])) if (f > 0).any() else 0
            rows = [r for r in range(n) if (hh[:, r] > t + 3).sum() >= max(2, (hh[:, r] > 0).sum() // 2)]
            if t >= 1 and rows and (best is None or lift > best[0]):
                best = (abs(lift), axis, flip, t, rows[0])
    if best is None or best[0] < 3:
        return None
    _, axis, flip, t, row = best
    # Work in a frame where depth runs along y, front at y=0.
    grid = target if axis == 1 else target.transpose(1, 0, 2)
    grid = grid[:, ::-1, :] if flip else grid
    X, Y, Z = grid.shape
    if row < 1 or row >= Y - 1:
        return None
    xs = np.nonzero(occ.any(axis=(1, 2)) if axis == 1 else occ.any(axis=(0, 2)))[0]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if not grid[x0:x1, row - 1, t - 1].all() or not grid[x0:x1, row, t + 1:].any(axis=1).all():
        return None                                   # nothing to hold the hinge from below / above
    # The hinge must be the only link between base and screen:
    shaped = grid.copy()
    shaped[:, :, t] = 0                               # the hinge layer holds only the two hinge halves
    shaped[:, row - 1, t + 1] = 0                     # nothing of the screen rests on the base's half
    shaped[x0:x1, row:, t - 1] = 0                    # nothing of the base holds up the screen's half
    shaped[x0:x1, row, t + 1] = np.where(shaped[x0:x1, row, t + 1] > 0, shaped[x0:x1, row, t + 1],
                                         COLOR_INDEX["dark_gray"])  # the screen's bottom edge on its hinge half
    shaped[x0:x1, row - 1:row + 1, t] = COLOR_INDEX["dark_gray"]
    # A row of real 1x2 click hinge pairs (44301 on the base, 44302 under the screen) along the hinge line.
    pivots = []
    for x in range(x0, x1 - 1, 2):
        pivots += [{"type": "click_a", "w": 2, "l": 1, "x": x, "y": row - 1, "z": t, "pin_x": x, "pin_y": row,
                    "pin_axis": "x", "color": "dark_gray"},
                   {"type": "click_b", "w": 2, "l": 1, "x": x, "y": row, "z": t, "pin_x": x, "pin_y": row,
                    "pin_axis": "x", "color": "dark_gray"}]
    shaped[x0 + 2 * ((x1 - x0) // 2):x1, row - 1:row + 1, t] = 0  # an odd last stud has no hinge
    body = build(shaped, attempts=attempts, given=structural(pivots), anchored=True)
    parts = _keep_connected(pivots + body.pieces[len(pivots):])
    return _verdict(parts, shaped, "lid")
