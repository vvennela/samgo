"""Rolling car with the wheels where the photo has them: the tire outlines in the body become real wheels.

The SAM 3D mesh already contains the car's tires, so the voxel body has dark tire outlines near the ground. This
builder finds them (center along the car, diameter), picks the real LEGO wheel closest in size, cuts a wheel arch
into the body around each one, and mounts it on a stop axle through Technic bricks inside the body. The body is
not lifted onto a separate chassis, and every part is a real LEGO part:
- stop axles only in real lengths (3, 4, 5, 8), through at least 2 aligned round holes, the stop on the outside;
- wheels from the measured LDraw sizes (technic.WHEELS), inside the body's width, under arches;
- Technic brick holes 5.6 mm above their bottom (LDraw), so each wheel's height follows from its brick row.
The reference answer for the running-gear training task (spec.md section 0).
"""

from dataclasses import dataclass

import numpy as np

from .builder import build
from .parts import COLOR_INDEX
from .stability import stability
from .technic import (AXLE_LENGTHS, HOLE_HEIGHT, PLATE, STOP_AXLE_LENGTHS, STUD, WHEELS, structural, verify_function,
                      wheel_studs)
from .verifier import MatchConfig, Report, verify

DARK = {COLOR_INDEX["black"], COLOR_INDEX["dark_gray"]}
ARCH_CLEARANCE = 1.6  # mm of air between a tire and its arch
SYMMETRIC = 0.85      # a cross-section this close to its mirror image is treated as symmetric
CHASSIS_COLOR = "black"


@dataclass
class Wheel:
    edge: int        # stud edge along the length where the axle sits
    size: str        # key of technic.WHEELS
    z: int           # plate row of the Technic bricks that hold its axle


@dataclass
class ArchedCar:
    parts: list[dict]
    target: np.ndarray
    wheels: list[Wheel]
    structure: Report
    function: Report
    symmetric: float = 0.0   # share of cross-sections made symmetric


def smooth(target: np.ndarray) -> np.ndarray:
    """Remove stray voxels (at most one face neighbour) and fill one-voxel holes (5+ face neighbours)."""
    occ = target > 0
    pad = np.pad(occ, 1)
    neighbours = sum(np.roll(pad, s, axis=a)[1:-1, 1:-1, 1:-1].astype(int) for a in range(3) for s in (1, -1))
    out = target.copy()
    out[occ & (neighbours <= 1)] = 0
    holes = ~occ & (neighbours >= 5)
    if holes.any():
        padc = np.pad(target, 1)
        for x, y, z in zip(*np.nonzero(holes)):
            around = [padc[x + 1 + dx, y + 1 + dy, z + 1 + dz] for dx, dy, dz in
                      ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
            around = [c for c in around if c]
            out[x, y, z] = max(set(around), key=around.count)
    return out


def symmetrize(target: np.ndarray) -> tuple[np.ndarray, float]:
    """Mirror each cross-section along the length that is nearly symmetric across the car's center line.

    SAM 3D guesses the far side from one photo, so a car comes out lopsided. A cross-section whose cells agree
    with their mirror image at least SYMMETRIC of the time is made symmetric (union of the two sides, colors
    from the fuller side); one that really differs (an open door, a spare wheel) is kept as it is. Returns the
    result and the share of cross-sections that were symmetric.
    """
    X, Y, Z = target.shape
    occ = target > 0

    def mirror(a: np.ndarray, shift: int) -> np.ndarray:
        """a flipped across the width, moved by `shift` cells (cells moved in from outside are empty)."""
        flipped = a[:, ::-1, :]
        out = np.zeros_like(a)
        if shift >= 0:
            out[:, shift:, :] = flipped[:, :Y - shift, :]
        else:
            out[:, :Y + shift, :] = flipped[:, -shift:, :]
        return out

    best = max(range(-2, 3), key=lambda k: (occ & mirror(occ, k)).sum())
    mirror_best = lambda a: mirror(a, best)  # noqa: E731
    out, mocc, mcol = target.copy(), mirror_best(occ), mirror_best(target)
    fuller_left = occ[:, : Y // 2].sum() >= occ[:, Y - Y // 2:].sum()
    symmetric = 0
    for x in range(X):
        union = occ[x] | mocc[x]
        if not union.any():
            continue
        if (occ[x] & mocc[x]).sum() / union.sum() < SYMMETRIC:
            continue
        symmetric += 1
        for y in range(Y):
            for z in range(Z):
                if not union[y, z]:
                    continue
                own, other = target[x, y, z], mcol[x, y, z]
                from_left = y < Y / 2
                out[x, y, z] = (own if own else other) if (from_left == fuller_left) else (other if other else own)
    return out, symmetric / max(int(occ.any(axis=(1, 2)).sum()), 1)


def find_tires(target: np.ndarray) -> list[tuple[int, int, float]]:
    """(first x, last x, outline diameter mm) of each tire, front to back.

    Only the tires touch the ground, so the runs of occupied columns in the body's lowest layer are the tires'
    contact patches. A tire's diameter is the larger of its length along the car and the height its dark
    outline reaches on the body's outer faces above that patch.
    """
    X, Y, Z = target.shape
    occupied = target > 0
    lowest = int(np.nonzero(occupied.any(axis=(0, 1)))[0].min())
    ground = occupied[:, :, lowest].any(axis=1)
    sides = np.concatenate([target[:, :2, :], target[:, Y - 2:, :]], axis=1)
    dark = np.isin(sides, list(DARK)).any(axis=1)  # x, z
    runs, start = [], None
    for x in range(X + 1):
        on = x < X and ground[x]
        if on and start is None:
            start = x
        if not on and start is not None:
            runs.append((start, x - 1))
            start = None
    tires = []
    for a, b in runs:
        if b - a + 1 > X // 3:          # a long flat bottom, not a tire
            continue
        lo, hi = max(a - 1, 0), min(b + 2, X)
        z = lowest
        while z + 1 < min(Z, lowest + int(Z * 0.6)) and dark[lo:hi, z + 1].any():
            z += 1
        diameter = max((b - a + 1) * STUD, (z - lowest + 1) * PLATE)
        tires.append((a, b, diameter))
    return tires


def pick_wheel(diameter: float, body_width: int) -> str:
    """The real wheel closest to the outline's diameter that leaves room for chassis between the two sides."""
    fits = [s for s in WHEELS if 2 * (wheel_studs(s) + 2) <= body_width] or ["30.4x14"]
    return min(fits, key=lambda s: abs(WHEELS[s][0] - diameter))


def _axle_row(radius: float) -> int:
    """Plate row of the Technic bricks so the wheel reaches about 3 mm below the body's bottom (the ground)."""
    return max(0, int(np.floor((radius - HOLE_HEIGHT - 1.0) / PLATE)))


def _build(target: np.ndarray, seed: int, attempts: int, beam: int) -> ArchedCar | None:
    """None if no two tire outlines are found or the car is too narrow for wheels and chassis."""
    if target.shape[1] > target.shape[0]:
        target = target.transpose(1, 0, 2)[::-1].copy()  # length along x (a rotation, not a mirror)
    target, sym = symmetrize(smooth(target))
    X, Y, Z = target.shape
    tires = find_tires(target)
    if len(tires) < 2:
        return None
    tires = [tires[0], tires[-1]]                     # front and rear
    occ = target > 0
    shaped = target.copy()
    chassis, gear, wheels = [], [], []
    for a, b, diameter in tires:
        # Wheels go against the body's sides where this wheel is (the lower body around the tire), centered on
        # it: the whole body can be wider on one side (mirrors, spoilers) and the grid wider still.
        local = np.nonzero(occ[max(a - 1, 0):b + 2, :, : max(3, Z // 2)].any(axis=(0, 2)))[0]
        Y0, Y1 = int(local.min()), int(local.max()) + 1
        W = Y1 - Y0
        size = pick_wheel(diameter, W)
        w = wheel_studs(size)
        r = WHEELS[size][0] / 2
        edge = int(round((a + b + 1) / 2))
        edge = min(max(edge, beam // 2), X - beam // 2)
        zb = _axle_row(r)
        centre = zb * PLATE + HOLE_HEIGHT
        through = min((L for L in AXLE_LENGTHS if W <= L <= 16), default=None)
        if through is not None and W - 2 * w >= 2:
            # One axle straight across, as small LEGO cars do: Technic bricks in every row between the wheels.
            for side in ("left", "right"):
                y_wheel = Y0 if side == "left" else Y1 - w
                for x in range(X):
                    for z in range(Z):
                        dx = max(x * STUD - edge * STUD, 0, edge * STUD - (x + 1) * STUD)
                        dz = max(z * PLATE - centre, 0, centre - (z + 1) * PLATE)
                        if dx * dx + dz * dz < (r + ARCH_CLEARANCE) ** 2:
                            shaped[x, y_wheel:y_wheel + w, z] = 0
                gear.append({"type": "wheel", "axis": "y", "x": edge, "z": zb, "y": y_wheel, "size": size})
            for y in range(Y0 + w, Y1 - w):
                shaped[edge - beam // 2:edge + beam // 2, y, zb:zb + 3] = COLOR_INDEX[CHASSIS_COLOR]
                chassis.append({"type": "technic_brick", "w": beam, "l": 1, "x": edge - beam // 2, "y": y, "z": zb,
                                "color": CHASSIS_COLOR, "hole": "round"})
            a0 = Y0 - (through - W) // 2
            gear.append({"type": "axle", "axis": "y", "x": edge, "z": zb, "y": a0, "length": through})
            # Bushes (3713) on any axle sticking out past a wheel, so each wheel is held on both sides.
            for y in list(range(a0, Y0)) + list(range(Y1, a0 + through)):
                gear.append({"type": "bush", "axis": "y", "x": edge, "z": zb, "y": y})
            wheels.append(Wheel(edge, size, zb))
            continue
        stop = min(L for L in STOP_AXLE_LENGTHS if L >= w + 2) if w + 2 <= max(STOP_AXLE_LENGTHS) else None
        if stop is None or 2 * stop > W:
            return None
        rows_in = stop - w                            # Technic rows the axle passes through inside the wheel
        for side in ("left", "right"):
            y_wheel = Y0 if side == "left" else Y1 - w
            rows = range(Y0 + w, Y0 + w + rows_in) if side == "left" else range(Y1 - w - rows_in, Y1 - w)
            # Arch: clear every body cell the tire could touch while turning, plus clearance.
            for x in range(X):
                for z in range(Z):
                    dx = max(x * STUD - edge * STUD, 0, edge * STUD - (x + 1) * STUD)
                    dz = max(z * PLATE - centre, 0, centre - (z + 1) * PLATE)
                    if dx * dx + dz * dz < (r + ARCH_CLEARANCE) ** 2:
                        shaped[x, y_wheel:y_wheel + w, z] = 0
            # Technic bricks (1x4, holes at edge-1, edge, edge+1) that hold the stop axle, in the body's place.
            for y in rows:
                shaped[edge - beam // 2:edge + beam // 2, y, zb:zb + 3] = COLOR_INDEX[CHASSIS_COLOR]
                chassis.append({"type": "technic_brick", "w": beam, "l": 1, "x": edge - beam // 2, "y": y, "z": zb,
                                "color": CHASSIS_COLOR, "hole": "round"})
            y0 = y_wheel if side == "left" else Y1 - w - rows_in
            gear.append({"type": "axle", "axis": "y", "x": edge, "z": zb, "y": y0, "length": stop, "stop": True})
            gear.append({"type": "wheel", "axis": "y", "x": edge, "z": zb, "y": y_wheel, "size": size})
        wheels.append(Wheel(edge, size, zb))
    # Nothing solid may sit on the axle line between the two sides at hole height: the stop axles end inside the
    # Technic rows, so only the rows themselves are on the line.
    def pull_on_wheels(pieces: list[dict]) -> float:
        peak = stability(pieces + gear, vehicle=True).metrics.get("peak_tension")
        return np.inf if peak is None else peak

    body = build(shaped, seed=seed, given=structural(chassis), pull=pull_on_wheels, attempts=attempts, anchored=True)
    parts = chassis + body.pieces[len(chassis):] + gear
    return ArchedCar(parts, shaped, wheels, verify(structural(parts), shaped, config=MatchConfig(),
                            anchors=frozenset(range(len(chassis)))),
                     verify_function(parts), sym)


def build_arched_car(target: np.ndarray, seed: int = 0, attempts: int = 2) -> ArchedCar | None:
    """The car on Technic beams 1x4, else 1x6, else 1x8 (longer beams spread the car's weight over more studs):
    the first that passes structure, function and stability, else the last one tried."""
    from .stability import stability

    car = None
    for beam in (4, 6, 8):
        car = _build(target, seed, attempts, beam)
        if car is None:
            return None
        if car.structure.passed and car.function.passed and stability(car.parts, vehicle=True).passed:
            return car
    return car
