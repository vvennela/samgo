"""S4f: algorithmic rolling-car builder. Car target grid + wheel positions -> a build that passes S3a and S3c.

The car's length runs along x. Two rows of Technic bricks form the chassis in the bottom brick layer,
cut into 1x8/6/4/2 so every axle position falls inside a brick (where the holes are), never on a seam.
The rest of that layer is left empty so the axles run free. A deck plate layer over both rows ties the
chassis into one frame; the body is the target from z=3 up, built by the S4 builder on top of it. Axles cross the car at the wheel positions with a wheel on
each side, just outside the body's width so nothing is in the wheels' way.
"""

from dataclasses import dataclass

import numpy as np

from .builder import build
from .parts import COLOR_INDEX
from .stability import stability
from .technic import TECHNIC_LENGTHS, structural, verify_function
from .verifier import MatchConfig, Report, verify

CHASSIS = 3          # plates: one brick layer
WHEEL_WIDTH = 2      # studs
MAX_AXLE = 16        # studs


@dataclass
class Car:
    parts: list[dict]
    target: np.ndarray       # what the structural parts are checked against (target with the chassis layer)
    structure: Report
    function: Report


def _segments(x0: int, x1: int, axle_edges: list[int]) -> list[tuple[int, int, bool]] | None:
    """Split [x0, x1) into (start, length, technic) pieces, fewest first, with no seam at an axle edge.

    Technic bricks come in 1x2/4/6/8; a 1x1 or 1x3 plain brick may fill a gap away from the axles.
    """
    best: dict[int, list] = {x0: []}
    for x in range(x0, x1):
        if x not in best:
            continue
        options = [(n, True) for n in TECHNIC_LENGTHS] + [(1, False), (3, False)]
        for n, technic in options:
            end = x + n
            if end > x1 or (end < x1 and end in axle_edges):
                continue
            plan = best[x] + [(x, n, technic)]
            if end not in best or len(plan) < len(best[end]):
                best[end] = plan
    plan = best.get(x1)
    if plan is None:
        return None
    # Every axle edge must be strictly inside a Technic segment.
    for e in axle_edges:
        if not any(technic and s < e < s + n for s, n, technic in plan):
            return None
    return plan


def build_car(target: np.ndarray, wheel_fractions: tuple[float, float] = (0.15, 0.85), seed: int = 0,
              chassis_color: str = "black", lift: bool = True) -> Car | None:
    """None if the target is too wide for an axle or no chassis split puts the axles inside Technic bricks.

    The car's length must run along x; a target that is longer in y is turned 90 degrees first.
    With `lift`, the whole shape sits on top of the chassis layer (one brick taller, nothing lost);
    without it, the chassis replaces the shape's bottom brick layer.
    """
    if target.shape[1] > target.shape[0]:
        target = target.transpose(1, 0, 2)[::-1].copy()  # a rotation, not a mirror
    if lift:
        target = np.concatenate([np.zeros(target.shape[:2] + (CHASSIS,), dtype=target.dtype), target], axis=2)
    X, Y, Z = target.shape
    occupied = np.nonzero(target.any(axis=(1, 2)))[0]
    x0, x1 = int(occupied.min()), int(occupied.max()) + 1
    if Y + 2 * WHEEL_WIDTH > MAX_AXLE or Z <= CHASSIS:
        return None
    edges = sorted({min(max(round(x0 + f * (x1 - x0)), x0 + 1), x1 - 1) for f in wheel_fractions})
    rows = [1, Y - 2] if Y >= 4 else [0, Y - 1]
    plan = _segments(x0, x1, edges)
    if plan is None:
        return None

    chassis = []
    for y in sorted(set(rows)):
        for s, n, technic in plan:
            piece = {"w": n, "l": 1, "x": s, "y": y, "z": 0, "color": chassis_color}
            chassis.append(piece | ({"type": "technic_brick", "hole": "round"} if technic else {"type": "brick"}))

    # Target for the structural build: the chassis rows in the bottom layer, the object above it.
    shaped = target.copy()
    shaped[:, :, :CHASSIS] = 0
    for p in chassis:
        shaped[p["x"]:p["x"] + p["w"], p["y"], :CHASSIS] = COLOR_INDEX[chassis_color]
    # Deck: one plate layer over both rows along the whole length, where the body leaves it empty.
    # Side-by-side bricks don't connect, so without it a chassis segment with no body above it floats free.
    deck = shaped[x0:x1, min(rows):max(rows) + 1, CHASSIS]
    deck[deck == 0] = COLOR_INDEX[chassis_color]
    running_gear = []
    for e in edges:
        y0 = -WHEEL_WIDTH
        running_gear.append({"type": "axle", "axis": "y", "x": e, "z": 0, "y": y0, "length": Y + 2 * WHEEL_WIDTH})
        running_gear += [{"type": "wheel", "axis": "y", "x": e, "z": 0, "y": y, "width": WHEEL_WIDTH} for y in (y0, Y)]

    def pull_on_axles(pieces: list[dict]) -> float:  # rank body layouts by stress with the car on its wheels
        peak = stability(chassis + pieces[len(chassis):] + running_gear, vehicle=True).metrics.get("peak_tension")
        return np.inf if peak is None else peak

    body = build(shaped, seed=seed, given=structural(chassis), pull=pull_on_axles)
    parts = chassis + body.pieces[len(chassis):] + running_gear
    return Car(parts, shaped, verify(structural(parts), shaped, config=MatchConfig()), verify_function(parts))
