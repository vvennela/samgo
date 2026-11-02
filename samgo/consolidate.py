"""Fewer, bigger pieces: merge neighbouring plates of one color on one layer into real larger plates.

A merged plate grips every stud its halves did, so nothing comes loose; plates merge only inside one rigid body
(never a door into the car, never across a hinge), glass only into sizes LEGO made in trans-clear. Repeated until
no pair merges, largest result first, so 1x1 fillers grow into 1x2 ... 2x8 where the layout allows.
"""

from .parts import FOOTPRINTS

VALID = {tuple(sorted(f)) for f in FOOTPRINTS}


def _merged(p: dict, q: dict) -> dict | None:
    """The single plate covering exactly p and q, if they form a real rectangle side by side."""
    if p["z"] != q["z"] or p.get("color") != q.get("color"):
        return None
    if p["y"] == q["y"] and p["l"] == q["l"] and (p["x"] + p["w"] == q["x"] or q["x"] + q["w"] == p["x"]):
        r = dict(p, x=min(p["x"], q["x"]), w=p["w"] + q["w"])
    elif p["x"] == q["x"] and p["w"] == q["w"] and (p["y"] + p["l"] == q["y"] or q["y"] + q["l"] == p["y"]):
        r = dict(p, y=min(p["y"], q["y"]), l=p["l"] + q["l"])
    else:
        return None
    return r if tuple(sorted((r["w"], r["l"]))) in VALID else None


def consolidate(parts: list[dict]) -> tuple[list[dict], int]:
    """(parts with neighbouring same-color plates merged inside each rigid body, how many merges)."""
    from .glass import GLASS
    from .kinematics import analyse

    def made(p):
        if p.get("color") != GLASS:
            return True
        from .colors import NAMED, exists
        from .parts import LDRAW_IDS

        pid = LDRAW_IDS.get(("plate", *sorted((p["w"], p["l"]))))
        return pid is not None and exists(pid, NAMED[GLASS])

    parts = [dict(p) for p in parts]
    merges = 0
    while True:
        bodies, _ = analyse(parts)
        body_of = {i: b for b, members in bodies.items() for i in members}
        plates = [i for i, p in enumerate(parts) if p["type"] == "plate"]
        by_layer: dict = {}
        for i in plates:
            by_layer.setdefault((parts[i]["z"], parts[i].get("color")), []).append(i)
        best = None
        for group in by_layer.values():
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    i, j = group[a], group[b]
                    if body_of.get(i) != body_of.get(j):
                        continue
                    r = _merged(parts[i], parts[j])
                    if r is None or not made(r):
                        continue
                    gain = (min(parts[i]["w"] * parts[i]["l"], parts[j]["w"] * parts[j]["l"]), -(r["w"] * r["l"]))
                    if best is None or gain < best[0]:  # smallest piece first, into the biggest plate
                        best = (gain, i, j, r)
        if best is None:
            # three identical plates stacked in one rigid body are one brick (same studs on top, same grip below)
            at = {}
            for i in plates:
                p = parts[i]
                at[(p["x"], p["y"], p["w"], p["l"], p.get("color"), p["z"])] = i
            for (x, y, w, l, color, z), i in sorted(at.items(), key=lambda kv: kv[0][5]):
                j, k = at.get((x, y, w, l, color, z + 1)), at.get((x, y, w, l, color, z + 2))
                if j is None or k is None or color == GLASS:
                    continue
                if not (body_of.get(i) == body_of.get(j) == body_of.get(k)):
                    continue
                brick = dict(parts[i], type="brick")
                parts = [p for n, p in enumerate(parts) if n not in (i, j, k)] + [brick]
                merges += 2
                break
            else:
                return parts, merges
            continue
        _, i, j, r = best
        parts = [p for k, p in enumerate(parts) if k not in (i, j)] + [r]
        merges += 1
