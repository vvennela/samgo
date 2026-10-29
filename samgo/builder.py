"""S4: algorithmic builder. Target grid -> build that passes the verifier.

Layers are filled from the ground up. In each layer the first unfilled cell,
in a scan order that changes per layer, becomes the corner of the best part
that fits there in one color. Scores favor large parts, parts resting on
something, and parts that join separate structures below (staggered seams),
and penalize stacking exactly on one part. Several randomized attempts run
and the best verified one wins: among those that pass, the one needing the least stud pull (S3b). Repair then drops pieces that still float or
sit in a smaller separate structure, as long as the build stays on target.
"""

import argparse
import json
from dataclasses import dataclass

import numpy as np

from .parts import COLOR_INDEX, FOOTPRINTS, HEIGHTS
from .stability import stability
from .verifier import MatchConfig, Report, verify

COLOR_NAMES = {i: name for name, i in COLOR_INDEX.items()}
# Both orientations of every footprint, largest first.
SHAPES = sorted({s for a, b in FOOTPRINTS for s in ((a, b), (b, a))}, key=lambda s: (-s[0] * s[1], s))
SCANS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]


@dataclass
class Build:
    pieces: list[dict]
    report: Report
    attempt: int
    pull: float = float("inf")  # S3b peak stud tension of this layout (lower is sturdier)


class _Layout:
    def __init__(self, target: np.ndarray):
        self.target = target
        self.owner = np.full(target.shape, -1, dtype=np.int32)
        self.pieces: list[dict] = []
        self.parent: list[int] = []
        self.grounded: list[bool] = []

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
            self.grounded[rb] = self.grounded[rb] or self.grounded[ra]

    def below(self, x: int, y: int, w: int, l: int, z: int) -> np.ndarray:
        if z == 0:
            return np.empty(0, dtype=np.int32)
        ids = self.owner[x:x + w, y:y + l, z - 1]
        return np.unique(ids[ids >= 0])

    def place(self, w: int, l: int, kind: str, x: int, y: int, z: int, color: int) -> None:
        i = len(self.pieces)
        self.owner[x:x + w, y:y + l, z:z + HEIGHTS[kind]] = i
        self.pieces.append({"w": w, "l": l, "type": kind, "x": x, "y": y, "z": z, "color": COLOR_NAMES[color]})
        self.parent.append(i)
        self.grounded.append(z == 0)
        # Anything occupying the layer just below is a piece whose top face is z.
        for j in self.below(x, y, w, l, z):
            self.union(i, int(j))


def _fits(layout: _Layout, x: int, y: int, z: int, w: int, l: int, h: int, color: int) -> bool:
    X, Y, Z = layout.target.shape
    if x < 0 or y < 0 or x + w > X or y + l > Y or z + h > Z:
        return False
    cells = (slice(x, x + w), slice(y, y + l), slice(z, z + h))
    return bool((layout.target[cells] == color).all() and (layout.owner[cells] < 0).all())


def _score(layout: _Layout, x: int, y: int, z: int, w: int, l: int, h: int) -> float:
    ids = layout.below(x, y, w, l, z)
    roots = {layout.find(int(j)) for j in ids}
    supported = z == 0 or len(ids) > 0
    grounded = z == 0 or any(layout.grounded[r] for r in roots)
    joins = max(len(roots) - 1, 0)
    # Crossing seams between pieces that are already joined still adds redundant
    # links, which later layers need when greedy joins happen in the wrong place.
    crossings = max(len(ids) - 1, 0)
    aligned = False
    if len(ids) == 1:
        p = layout.pieces[int(ids[0])]
        aligned = (p["x"], p["y"], p["w"], p["l"]) == (x, y, w, l)
    return w * l * (1 + 0.3 * (h == 3)) + 8 * joins + 3 * crossings + 4 * supported + 4 * grounded - 6 * aligned


def _layout(target: np.ndarray, rng: np.random.Generator | None, bricks: bool, given: list[dict] = ()) -> list[dict]:
    layout = _Layout(target)
    for p in given:  # pre-placed pieces (e.g. a car's chassis) are built around, never changed
        layout.place(p["w"], p["l"], p["type"], p["x"], p["y"], p["z"], COLOR_INDEX[p["color"]])
    X, Y, Z = target.shape
    kinds = ["brick", "plate"] if bricks else ["plate"]
    for z in range(Z):
        sx, sy = SCANS[z % 4] if rng is None else SCANS[rng.integers(4)]
        xs = range(X) if sx > 0 else range(X - 1, -1, -1)
        ys = range(Y) if sy > 0 else range(Y - 1, -1, -1)
        for ay in ys:
            for ax in xs:
                color = target[ax, ay, z]
                if color == 0 or layout.owner[ax, ay, z] >= 0:
                    continue
                best, best_score = None, -np.inf
                for w, l in SHAPES:
                    # The anchor is the first unfilled cell in scan order, so it is a corner.
                    x = ax if sx > 0 else ax - w + 1
                    y = ay if sy > 0 else ay - l + 1
                    for kind in kinds:
                        h = HEIGHTS[kind]
                        if not _fits(layout, x, y, z, w, l, h, color):
                            continue
                        s = _score(layout, x, y, z, w, l, h)
                        if rng is not None:
                            s += rng.uniform(0, 2)
                        if s > best_score:
                            best, best_score = (w, l, kind, x, y), s
                w, l, kind, x, y = best  # a 1x1 plate always fits the anchor
                layout.place(w, l, kind, x, y, z, int(color))
    return layout.pieces


def _repair(pieces: list[dict], target: np.ndarray, config: MatchConfig, keep: int = 0) -> tuple[list[dict], Report]:
    report = verify(pieces, target, config=config)
    drop = {i for issue in report.issues if issue.code in ("floating", "disconnected") for i in issue.pieces if i >= keep}
    if not drop or "overlap" in report.codes:
        return pieces, report
    kept = [p for i, p in enumerate(pieces) if i not in drop]
    return kept, verify(kept, target, config=config)


def build(target: np.ndarray, attempts: int = 8, seed: int = 0, config: MatchConfig = MatchConfig(),
          given: list[dict] = (), pull=None) -> Build:
    """Return the best of several layouts: passing first, then least stud pull, then highest IoU, then fewest pieces.

    `given` pieces come first in the result and are kept as they are; the target must contain their cells.
    `pull(pieces)` measures stress (default: S3b peak tension standing on the ground; inf if it can't balance).
    """
    pull = pull or _ground_pull
    rng = np.random.default_rng(seed)
    best: Build | None = None
    for attempt in range(attempts):
        # Attempt 0 is deterministic; later ones randomize scans and ties, and some use plates only.
        pieces = _layout(target, None if attempt == 0 else rng, bricks=attempt % 4 != 3, given=list(given))
        pieces, report = _repair(pieces, target, config, keep=len(given))
        candidate = Build(pieces, report, attempt, pull(pieces) if report.passed else np.inf)
        if best is None or _rank(candidate) > _rank(best):
            best = candidate
    return best


def _ground_pull(pieces: list[dict]) -> float:
    peak = stability(pieces).metrics.get("peak_tension")
    return np.inf if peak is None else peak


def _rank(b: Build) -> tuple:
    return (b.report.passed, -b.pull, b.report.metrics.get("iou", 0.0), -len(b.pieces))


def main() -> None:
    from .targets import FAMILIES, sample

    parser = argparse.ArgumentParser(description="Pass rate of the algorithmic builder on procedural targets")
    parser.add_argument("-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stats = {f: {"n": 0, "pass": 0, "fails": {}} for f in FAMILIES}
    for k in range(args.n):
        family, target = sample(args.seed + k)
        result = build(target, seed=k)
        s = stats[family]
        s["n"] += 1
        s["pass"] += result.report.passed
        for code in result.report.codes:
            s["fails"][code] = s["fails"].get(code, 0) + 1
    total = sum(s["pass"] for s in stats.values())
    for f, s in stats.items():
        if s["n"]:
            print(f"{f:10s} {s['pass']:3d}/{s['n']:<3d} {s['fails'] or ''}")
    print(json.dumps({"pass_rate": total / args.n, "n": args.n}))


if __name__ == "__main__":
    main()
