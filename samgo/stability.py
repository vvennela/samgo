"""S3b: does the build stand? Static force balance as a linear program (StableLego-style).

Every piece must be in equilibrium: vertical forces and the torques about x and y sum to zero, with
its weight (proportional to volume) at its center. Unknowns are vertical forces at contact points:
four around the rim of each shared stud (so one stud can resist tipping by pushing on one side and
pulling on the other), compression unlimited, tension (one piece pulling another up) limited by
stud clutch, split over the four points. Supports push up only: the ground under every z = 0 stud, or, for a vehicle,
the axles where they pass through the chassis holes. The LP finds the force pattern that needs the
least tension; the build is stable if that peak tension is within CLUTCH, and each piece's score is
its own peak tension as a fraction of CLUTCH.

Units: one plate-sized cell weighs 1 (a 1x1 plate is about 0.4 g). CLUTCH is per stud, in those units:
one stud holds roughly 1-3 N, i.e. 250-750 cells, and the default takes the low end. It is an estimate
to calibrate against a physical build (S10). Per-piece scores come from the min-peak solution, which
spreads tension evenly, so they show where load is carried more than which single joint is weakest.
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .parts import HEIGHTS
from .technic import _holes
from .verifier import Issue, Report

CLUTCH = 250.0
STUD_RADIUS = 0.3   # studs (2.4 mm of 8 mm)
RIM = [(-STUD_RADIUS, -STUD_RADIUS), (-STUD_RADIUS, STUD_RADIUS), (STUD_RADIUS, -STUD_RADIUS), (STUD_RADIUS, STUD_RADIUS)]


def stability(parts: list[dict], vehicle: bool = False, clutch: float = CLUTCH) -> Report:
    """Report with metrics peak_tension, score per piece (index into the solid pieces), and the verdict."""
    solid = [p for p in parts if isinstance(p, dict) and p.get("type") in ("plate", "brick", "technic_brick")]
    if not solid:
        return Report(False, [Issue("empty", "nothing to stand")], {})
    owner_top: dict[tuple[int, int, int], int] = {}   # (x, y, z of top face) -> piece
    for i, p in enumerate(solid):
        h = HEIGHTS["brick" if p["type"] == "technic_brick" else p["type"]]
        for x in range(p["x"], p["x"] + p["w"]):
            for y in range(p["y"], p["y"] + p["l"]):
                owner_top[(x, y, p["z"] + h)] = i

    # Variables: one force per contact point, then tau (the peak tension). Contacts: (piece above, piece below or
    # None for a support, x, y, is_support).
    contacts = []
    for i, p in enumerate(solid):
        for x in range(p["x"], p["x"] + p["w"]):
            for y in range(p["y"], p["y"] + p["l"]):
                if (below := owner_top.get((x, y, p["z"]))) is not None:
                    contacts += [(i, below, x + 0.5 + dx, y + 0.5 + dy, False) for dx, dy in RIM]
                elif p["z"] == 0 and not vehicle:  # the ground pushes up across the whole cell
                    contacts += [(i, None, x + cx, y + cy, True) for cx in (0, 1) for cy in (0, 1)]
    if vehicle:
        axle_edges = defaultdict(set)
        for a in parts:
            if isinstance(a, dict) and a.get("type") == "axle":
                axle_edges[(a["axis"], a["x"] if a["axis"] == "y" else a["y"], a["z"])].add(True)
        for i, p in enumerate(solid):
            if p["type"] != "technic_brick":
                continue
            for axis, edge_mm, _, s0, s1 in _holes(p):
                edge = round(edge_mm / 8.0)
                if (axis, edge, p["z"]) in axle_edges:  # the axle bears on the whole hole, wall to wall
                    for s in (s0 / 8.0, s1 / 8.0):
                        contacts.append((i, None, edge, s, True) if axis == "y" else (i, None, s, edge, True))

    n, m = len(contacts), len(solid)
    tau = n
    a_eq = lil_matrix((3 * m, n + 1))
    b_eq = np.zeros(3 * m)
    centers = []
    for i, p in enumerate(solid):
        h = HEIGHTS["brick" if p["type"] == "technic_brick" else p["type"]]
        centers.append((p["x"] + p["w"] / 2, p["y"] + p["l"] / 2))
        b_eq[3 * i] = p["w"] * p["l"] * h  # weight; torques of the weight about the center are zero
    for k, (up, down, x, y, _) in enumerate(contacts):
        for piece, sign in ((up, 1.0), (down, -1.0)):  # +F lifts the upper piece, pushes the lower one down
            if piece is None:
                continue
            cx, cy = centers[piece]
            a_eq[3 * piece, k] += sign
            a_eq[3 * piece + 1, k] += sign * (x - cx)
            a_eq[3 * piece + 2, k] += sign * (y - cy)
    # Tension bound: F >= -tau for stud contacts, F >= 0 for supports.
    a_ub = lil_matrix((n, n + 1))
    for k, (_, _, _, _, support) in enumerate(contacts):
        if not support:
            a_ub[k, k], a_ub[k, tau] = -1.0, -1.0
    bounds = [(0, None) if c[4] else (None, None) for c in contacts] + [(0, None)]
    cost = np.zeros(n + 1)
    cost[tau] = 1.0
    res = linprog(cost, A_ub=a_ub.tocsr(), b_ub=np.zeros(n), A_eq=a_eq.tocsr(), b_eq=b_eq, bounds=bounds,
                  method="highs")
    if res.status != 0:
        return Report(False, [Issue("unbalanced", "no set of stud forces holds every piece up (it would tip or fall)")],
                      {"peak_tension": None})
    forces = res.x[:n]
    peak = float(res.x[tau])
    per_piece = np.zeros(m)
    for (up, down, _, _, support), f in zip(contacts, forces):
        if not support and f < 0:
            for piece in (up, down):
                per_piece[piece] = max(per_piece[piece], -f * len(RIM) / clutch)
    metrics = {"peak_tension": round(peak, 3), "clutch": clutch, "max_score": round(float(per_piece.max()), 3),
               "scores": [round(float(s), 3) for s in per_piece]}
    if peak > clutch / len(RIM):
        worst = [int(i) for i in np.nonzero(per_piece > 1.0)[0]]
        return Report(False, [Issue("overloaded", f"needs {peak * len(RIM):.1f} units of pull on one stud; clutch "
                                                  f"holds {clutch}", worst)], metrics)
    return Report(True, [], metrics)
