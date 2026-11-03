"""Fewer, bigger pieces, same geometry: neighbours that together are exactly one piece of our collection become it.

- Beside: two plates (or two bricks) of one color on one layer whose union is a real rectangle become that plate
  (brick). Repeated until nothing merges, smallest pieces first, so 1x1 + 1x1 -> 1x2, 1x2 + 1x2 -> 2x2 ... 2x8.
- Above/below: three plates of one footprint and color stacked directly on each other are one brick.

The cells filled and their colors never change. Pieces merge only inside one rigid body (never a door into the car,
never across a hinge or onto a turning axle), so every stud grip is kept: a merged piece grips every stud its parts
did. A merged piece must be a size LEGO made in that color (glass included). Mechanism parts are never touched.
"""

from .parts import FOOTPRINTS

VALID = {tuple(sorted(f)) for f in FOOTPRINTS}
SOLID = ("plate", "brick")


def _merged(p: dict, q: dict) -> dict | None:
    """The single piece covering exactly p and q, if they form a real rectangle side by side (same type)."""
    if p["type"] != q["type"] or p["z"] != q["z"] or p.get("color") != q.get("color"):
        return None
    if p["y"] == q["y"] and p["l"] == q["l"] and (p["x"] + p["w"] == q["x"] or q["x"] + q["w"] == p["x"]):
        r = dict(p, x=min(p["x"], q["x"]), w=p["w"] + q["w"])
    elif p["x"] == q["x"] and p["w"] == q["w"] and (p["y"] + p["l"] == q["y"] or q["y"] + q["l"] == p["y"]):
        r = dict(p, y=min(p["y"], q["y"]), l=p["l"] + q["l"])
    else:
        return None
    return r if tuple(sorted((r["w"], r["l"]))) in VALID else None


def _made(p: dict) -> bool:
    """LEGO produced this size in this color (unknown colors are left to the later color legalizer)."""
    from .colors import NAMED, exists
    from .parts import LDRAW_IDS

    pid = LDRAW_IDS.get((p["type"], *sorted((p["w"], p["l"]))))
    if pid is None:
        return False
    color = NAMED.get(p.get("color"))
    if color is None:
        return True
    try:
        return exists(pid, color)
    except Exception:  # noqa: BLE001 - no catalogue entry: keep the original pieces
        return False


def consolidate(parts: list[dict]) -> tuple[list[dict], int]:
    """(parts with neighbours merged into single real pieces inside each rigid body, how many pieces were saved)."""
    from .kinematics import analyse

    parts = [dict(p) for p in parts]
    bodies, _ = analyse(parts)
    body = [None] * len(parts)
    for b, members in (bodies.items() if isinstance(bodies, dict) else enumerate(bodies)):
        for i in members:
            body[i] = b
    keep = [p for i, p in enumerate(parts) if p["type"] not in SOLID]
    solid = [(p, body[i]) for i, p in enumerate(parts) if p["type"] in SOLID]
    saved = 0
    changed = True
    while changed:
        changed = False
        # beside: index every piece by the edge a partner would share with it
        alive = {k: s for k, s in enumerate(solid)}
        by_edge: dict = {}
        for k, (p, b) in alive.items():
            by_edge.setdefault(("x", p["type"], p["z"], p.get("color"), b, p["y"], p["l"], p["x"] + p["w"]), []).append(k)
            by_edge.setdefault(("y", p["type"], p["z"], p.get("color"), b, p["x"], p["w"], p["y"] + p["l"]), []).append(k)
        order = sorted(alive, key=lambda k: alive[k][0]["w"] * alive[k][0]["l"])
        used = set()
        out = []
        for k in order:
            if k in used:
                continue
            p, b = alive[k]
            best = None
            for key in (("x", p["type"], p["z"], p.get("color"), b, p["y"], p["l"], p["x"]),
                        ("y", p["type"], p["z"], p.get("color"), b, p["x"], p["w"], p["y"])):
                for j in by_edge.get(key, []):  # pieces ending where p starts
                    if j in used or j == k:
                        continue
                    r = _merged(alive[j][0], p)
                    if r is not None and _made(r) and (best is None or r["w"] * r["l"] > best[1]["w"] * best[1]["l"]):
                        best = (j, r)
            if best is None:
                continue
            j, r = best
            used |= {j, k}
            out.append((r, b))
            saved += 1
            changed = True
        solid = [s for k, s in alive.items() if k not in used] + out
        if changed:
            continue
        # above/below: three identical plates stacked in one rigid body are one brick
        at = {(p["x"], p["y"], p["w"], p["l"], p.get("color"), b, p["z"]): k
              for k, (p, b) in enumerate(solid) if p["type"] == "plate"}
        used = set()
        out = []
        for (x, y, w, l, color, b, z), k in sorted(at.items(), key=lambda kv: kv[0][6]):
            j, m = at.get((x, y, w, l, color, b, z + 1)), at.get((x, y, w, l, color, b, z + 2))
            if j is None or m is None or {k, j, m} & used:
                continue
            brick = dict(solid[k][0], type="brick")
            if not _made(brick):
                continue
            used |= {k, j, m}
            out.append((brick, b))
            saved += 2
            changed = True
        solid = [s for k, s in enumerate(solid) if k not in used] + out
    return keep + [p for p, _ in solid], saved
