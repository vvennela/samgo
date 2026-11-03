"""Mirror an edit list across a model's left-right mirror plane (the middle of y, where samgo.voxelize puts it).

Used when Spark says two mechanisms are mirror twins (a left and a right aileron): Glimmer builds one, code mirrors
Glimmer's verified edits, and the checker verifies the mirrored copy like any answer. Any part type: the mirrored
position is found from the part's extent in y (its box, or its axle line), so plates, hinge plates, Technic bricks,
axles, bushes, liftarms and wheels all mirror the same way.
"""

from .kinematics import STUD, box

FLIP = {"+y": "-y", "-y": "+y"}


def _y_extent(p: dict) -> tuple[float, float]:
    try:
        lo, hi = box(p)
        return float(lo[1]), float(hi[1])
    except Exception:  # noqa: BLE001 - axles and bushes: their line
        from .technic import _line

        axis, edge, _, start, end = _line(p)
        return (float(start), float(end)) if axis == "y" else (float(edge), float(edge))


def mirror_part(p: dict, width: int) -> dict | None:
    """The part mirrored across y = width / 2 studs, or None if no position mirrors it exactly."""
    q = dict(p)
    if q.get("facing") in FLIP:
        q["facing"] = FLIP[q["facing"]]
    if q.get("type") == "liftarm" and q.get("axis") == "y" and q.get("axle_hole") in ("start", "end"):
        q["axle_hole"] = {"start": "end", "end": "start"}[q["axle_hole"]]
    lo, hi = _y_extent(p)
    want = (width * STUD - hi, width * STUD - lo)
    for y in range(-8, width + 9):
        q["y"] = y
        try:
            a, b = _y_extent(q)
        except Exception:  # noqa: BLE001
            return None
        if abs(a - want[0]) < 0.01 and abs(b - want[1]) < 0.01:
            return q
    return None


def mirror_edits(edits: list[dict], body: list[dict], width: int) -> list[dict] | None:
    """The edit list mirrored (indices in the whole model). None if a part it changes has no exact mirror twin."""
    where = {}
    for i, p in enumerate(body):
        where.setdefault(tuple(sorted((k, str(v)) for k, v in p.items())), i)

    def twin(i):
        m = mirror_part(body[i], width)
        return None if m is None else where.get(tuple(sorted((k, str(v)) for k, v in m.items())))

    out = []
    for e in edits:
        if e["op"] == "clear":
            x0, x1, y0, y1, z0, z1 = e["box"]
            out.append({"op": "clear", "box": [x0, x1, width - 1 - y1, width - 1 - y0, z0, z1]})
            continue
        f = dict(e)
        if "index" in e:
            j = twin(e["index"])
            if j is None:
                return None
            f["index"] = j
        if "part" in e:
            m = mirror_part(e["part"], width)
            if m is None:
                return None
            f["part"] = m
        out.append(f)
    return out
