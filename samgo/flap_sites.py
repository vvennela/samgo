"""Where a flap can be cut out of a thin structure: every strip along its +x edge that a plain answer turns into a
working flap (clear a channel on three sides, bar and clip plates on the local top at the hinge line) and that the
generic checks pass - nothing comes loose, the strip swings clear. Code verifies the handoff before anyone is asked
to build it; which of the sites to use is a choice (Spark's, or the caller's preference).

Works on any body and any set of its parts (a wing, a tail, a car's spoiler); nothing here knows what the object is.
"""

from .edits import EditError, apply_edits
from .free_edit_test import check
from .primitives import occupancy

FLAP = [{"name": "flap", "kind": "swing", "axis": "horizontal", "degrees": 20, "count": 1}]


def site_answer(body: list[dict], site: dict, color: str = "dark_gray") -> list[dict]:
    """The plain edit list for a site (global indices): the three channel clears, then the hinge plates."""
    xh, y0, y1, z = site["xh"], site["y0"], site["y1"], site["z"]
    edge = max(p["x"] + p["w"] for p in body)
    edits = [{"op": "clear", "box": [xh, xh, y0, y1, 0, 60]},
             {"op": "clear", "box": [xh, edge, y0 - 1, y0 - 1, 0, 60]},
             {"op": "clear", "box": [xh, edge, y1 + 1, y1 + 1, 0, 60]}]
    for y in site["hinge_rows"]:
        edits += [{"op": "add", "part": dict(type="bar_plate", w=2, l=1, x=xh - 2, y=y, z=z, facing="+x", color=color)},
                  {"op": "add", "part": dict(type="clip_plate", w=2, l=1, x=xh + 1, y=y, z=z, facing="-x",
                                             color=color)}]
    return edits


def sites(body: list[dict], part_ids: list[int], lengths=(3, 4, 5), chords=(2, 3, 4), rng=None,
          limit: int | None = None, ys: tuple[int, int] | None = None, crop: bool = False) -> list[dict]:
    """Verified flap sites on the +x edge of the parts `part_ids` of `body`: all of them, or (with `rng` and
    `limit`) the first `limit` found in a random order."""
    edge_x: dict = {}
    for i in part_ids:
        p = body[i]
        for y in range(p["y"], p["y"] + p["l"]):
            edge_x[y] = max(edge_x.get(y, -1), p["x"] + p["w"])
    occ = occupancy(body)
    found = []
    tries = [(L, y0, chord) for L in lengths for y0 in sorted(edge_x) for chord in chords
             if ys is None or (y0 + L - 1 >= ys[0] and y0 <= ys[1])]  # `ys`: only spans meeting these rows
    if rng is not None:
        rng.shuffle(tries)
    for L, y0, chord in tries:
        if limit is not None and len(found) >= limit:
            break
        span = list(range(y0, y0 + L))
        if any(y not in edge_x for y in span):
            continue
        xh = min(edge_x[y] for y in span) - chord
        top = {y: max((z for (x, yy, z) in occ if yy == y and xh - 2 <= x <= xh + 2), default=-1)
               for y in span}
        pairs = [(a, b) for a in span for b in span if a < b and top[a] == top[b] >= 0]
        if not pairs:
            continue
        hr = max(pairs, key=lambda ab: ab[1] - ab[0])
        site = dict(xh=xh, y0=span[0], y1=span[-1], z=top[hr[0]] + 1, hinge_rows=list(hr), chord=chord)
        try:
            a = apply_edits(body, site_answer(body, site))
            v = check(body, a.parts, a.sources, FLAP, stand=False, crop=crop)
        except EditError:
            continue
        if all(v[k][0] for k in v if k in ("structure", "connected") or k.endswith("surfaces")):
            found.append(site)
    return found


def pick(found: list[dict], near_y: float) -> dict | None:
    """The biggest flap (span x chord), then the one nearest `near_y`."""
    return min(found, key=lambda s: (-(s["y1"] - s["y0"] + 1) * s["chord"],
                                     abs((s["y0"] + s["y1"]) / 2 - near_y)), default=None)
