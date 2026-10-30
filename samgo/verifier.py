"""Deterministic build verifier: the only judge of whether a build is valid.

A build passes when every piece is a real palette part, inside the grid, with
no overlaps, all pieces form one stud-connected structure that touches the
ground, and the result matches the target closely enough.

Connection rule: two pieces are connected when one sits directly on top of
the other (top face z + h == bottom face z) and their footprints share at
least one stud. Touching side by side is not a connection. This checks
connectivity, not clutch strength, balance, or build order.
"""

from dataclasses import dataclass, field

import numpy as np

from .parts import COLOR_INDEX, PartError, Piece, parse_piece

VERSION = "samgo-verifier-v1"


@dataclass(frozen=True)
class MatchConfig:
    # Product default; curriculum tasks use min_iou=1.0 (exact occupancy).
    min_iou: float = 0.85
    min_color_accuracy: float = 0.80


@dataclass
class Issue:
    code: str  # invalid_part, out_of_bounds, overlap, floating, disconnected, empty, off_target
    message: str
    pieces: list[int] = field(default_factory=list)


@dataclass
class Report:
    passed: bool
    issues: list[Issue]
    metrics: dict

    @property
    def codes(self) -> set[str]:
        return {i.code for i in self.issues}


def verify(raw_pieces: list, target: np.ndarray | None = None, bounds: tuple[int, int, int] | None = None,
           config: MatchConfig = MatchConfig(), anchors: frozenset[int] = frozenset()) -> Report:
    """Check a build. `target` is an (X, Y, Z) int grid of COLOR_INDEX values, 0 = empty.

    `anchors`: indices of pieces that count as supported even off the ground (a car's axle-holding Technic
    bricks, which stand on the wheels).

    Without a target, `bounds` limits the grid; without either, only z >= 0 and
    non-negative x/y are enforced and the target check is skipped.
    """
    issues: list[Issue] = []
    metrics: dict = {"verifier": VERSION, "pieces": len(raw_pieces) if isinstance(raw_pieces, list) else 0}

    if not isinstance(raw_pieces, list):
        return Report(False, [Issue("invalid_part", "build must be a list of pieces")], metrics)
    if not raw_pieces:
        return Report(False, [Issue("empty", "build has no pieces")], metrics)

    pieces: list[Piece] = []
    for i, raw in enumerate(raw_pieces):
        try:
            pieces.append(parse_piece(raw))
        except PartError as e:
            issues.append(Issue("invalid_part", f"piece {i}: {e}", [i]))
    if issues:
        return Report(False, issues, metrics)

    shape = target.shape if target is not None else bounds
    if shape is None:
        shape = tuple(max(s, 1) for s in (
            max(p.x + p.w for p in pieces), max(p.y + p.l for p in pieces), max(p.z + p.h for p in pieces)))
    outside = [i for i, p in enumerate(pieces)
               if p.x < 0 or p.y < 0 or p.z < 0
               or p.x + p.w > shape[0] or p.y + p.l > shape[1] or p.z + p.h > shape[2]]
    if outside:
        issues.append(Issue("out_of_bounds", f"{len(outside)} piece(s) outside the {shape} grid", outside))
        return Report(False, issues, metrics)

    count = np.zeros(shape, dtype=np.int32)
    owner = np.full(shape, -1, dtype=np.int32)
    colors = np.zeros(shape, dtype=np.int32)
    for i, p in enumerate(pieces):
        count[p.cells] += 1
        owner[p.cells] = i
        colors[p.cells] = COLOR_INDEX[p.color]

    overlapping = [i for i, p in enumerate(pieces) if count[p.cells].max() > 1]
    if overlapping:
        issues.append(Issue("overlap", f"{int((count > 1).sum())} cell(s) occupied twice", overlapping))

    components = _components(pieces, owner, shape)
    grounded = [c for c in components if any(pieces[i].z == 0 or i in anchors for i in c)]
    floating = sorted(i for c in components if c not in grounded for i in c)
    metrics["components"] = len(components)
    if floating:
        issues.append(Issue("floating", f"{len(floating)} piece(s) not supported from the ground", floating))
    if len(grounded) > 1:
        # Everything outside the largest grounded structure is reported.
        largest = max(grounded, key=len)
        rest = sorted(i for c in grounded if c is not largest for i in c)
        issues.append(Issue("disconnected", f"{len(grounded)} separate grounded structures", rest))

    if target is not None:
        built, wanted = count > 0, target > 0
        union = int((built | wanted).sum())
        both = built & wanted
        iou = int(both.sum()) / union if union else 1.0
        color_accuracy = float((colors[both] == target[both]).mean()) if both.any() else 0.0
        metrics.update(iou=iou, color_accuracy=color_accuracy)
        if iou < config.min_iou or color_accuracy < config.min_color_accuracy:
            issues.append(Issue("off_target", f"IoU {iou:.3f} (need {config.min_iou}), "
                                              f"color {color_accuracy:.3f} (need {config.min_color_accuracy})"))

    return Report(not issues, issues, metrics)


def _components(pieces: list[Piece], owner: np.ndarray, shape: tuple) -> list[list[int]]:
    parent = list(range(len(pieces)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(pieces):
        top = p.z + p.h
        if top >= shape[2]:
            continue
        for j in np.unique(owner[p.x:p.x + p.w, p.y:p.y + p.l, top]):
            if j >= 0 and pieces[j].z == top:
                parent[find(i)] = find(int(j))

    groups: dict[int, list[int]] = {}
    for i in range(len(pieces)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
