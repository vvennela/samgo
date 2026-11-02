"""Atomic edits on original part numbering, with provenance for checker feedback."""

from copy import deepcopy
from dataclasses import dataclass


class EditError(ValueError):
    pass


@dataclass
class AppliedEdits:
    parts: list[dict]
    sources: list[int | None]  # original full-build index; None for an added part
    changed: frozenset[int]    # candidate indices of added or replaced parts
    removed: frozenset[int]    # original full-build indices


H = {"plate": 1, "brick": 3}


def _clear(parts: list[dict], sources: list, box: list[int]) -> tuple[list[dict], list, set]:
    """Empty an inclusive box [x0, x1, y0, y1, z0, z1] in the whole model: parts inside go; a plate or brick that
    crosses the box's edge is cut there and its outside cells re-laid as real plates (the biggest piece keeps the
    part's identity). Returns the parts, their sources, and the positions of pieces the cut made."""
    from .primitives import tile_cells

    x0, x1, y0, y1, z0, z1 = box
    inside = lambda x, y, z: x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1
    out, src, made = [], [], set()
    for p, s in zip(parts, sources):
        if p.get("type") not in H or "w" not in p:
            if p.get("type") in ("bar_plate", "clip_plate", "technic_brick", "hinge_base", "hinge_top") and \
                    any(inside(x, y, z) for x in range(p["x"], p["x"] + p.get("w", 1))
                        for y in range(p["y"], p["y"] + p.get("l", 1)) for z in (p["z"],)):
                continue  # a connector part inside the box goes too
            out.append(p)
            src.append(s)
            continue
        layers = range(p["z"], p["z"] + H[p["type"]])
        hit = any(inside(x, y, z) for x in range(p["x"], p["x"] + p["w"]) for y in range(p["y"], p["y"] + p["l"])
                  for z in layers)
        if not hit:
            out.append(p)
            src.append(s)
            continue
        # what is left of the part: a brick the box spans top to bottom stays bricks; otherwise layer by layer,
        # as plates; the biggest piece keeps the part's identity
        foot = [(x, y) for x in range(p["x"], p["x"] + p["w"]) for y in range(p["y"], p["y"] + p["l"])]
        fewest = lambda cs, z=None: min((tile_cells(cs, 0), tile_cells(cs, 1)), key=len)
        pieces = []
        if p["type"] == "brick" and all(z0 <= z <= z1 for z in layers):
            rest = {(x, y) for x, y in foot if not (x0 <= x <= x1 and y0 <= y <= y1)}
            pieces = [dict(p, x=a, y=b, w=w, l=l) for a, b, w, l in fewest(rest)]
        else:
            for z in layers:
                rest = {(x, y) for x, y in foot if not inside(x, y, z)}
                pieces += [dict(p, type="plate", x=a, y=b, w=w, l=l, z=z) for a, b, w, l in fewest(rest, z)]
        pieces.sort(key=lambda q: -q["w"] * q["l"] * H[q["type"]])
        for k, q in enumerate(pieces):
            out.append(q)
            src.append(s if k == 0 else None)
            made.add(len(out) - 1)
    return out, src, made


def apply_edits(body: list[dict], edits: list[dict], local: list[int] | None = None,
                allowed_ops: frozenset[str] = frozenset({"set", "remove", "add", "clear"})) -> AppliedEdits:
    """Validate the complete edit list before constructing a detached candidate.

    Indices always refer to the original local view, never to added parts or shifted positions.
    Multiple operations on one original part are rejected, regardless of their ordering.
    This validates edit syntax; the assembly checker validates part geometry and function.
    """
    if not isinstance(body, list) or any(not isinstance(p, dict) for p in body):
        raise EditError("input build must be a list of part objects")
    if not isinstance(edits, list):
        raise EditError("edits must be a JSON list")
    local = list(range(len(body))) if local is None else local
    if any(type(i) is not int or not 0 <= i < len(body) for i in local) or len(set(local)) != len(local):
        raise EditError("invalid local part mapping")
    changes, added, boxes = {}, [], []
    fields = {"set": {"op", "index", "part"}, "remove": {"op", "index"}, "add": {"op", "part"}, "clear": {"op", "box"}}
    for n, e in enumerate(edits):
        if not isinstance(e, dict) or not isinstance(e.get("op"), str) or e["op"] not in fields:
            raise EditError(f"edit {n}: operation must be set, remove, add or clear")
        op = e["op"]
        if op not in allowed_ops:
            raise EditError(f"edit {n}: {op} is not allowed for this task; use {', '.join(sorted(allowed_ops))}")
        if set(e) != fields[op]:
            raise EditError(f"edit {n}: {op} requires exactly {', '.join(sorted(fields[op]))}")
        if op == "clear":
            b = e["box"]
            if not (isinstance(b, list) and len(b) == 6 and all(type(v) is int for v in b)
                    and b[0] <= b[1] and b[2] <= b[3] and b[4] <= b[5]):
                raise EditError(f"edit {n}: clear box must be [x0, x1, y0, y1, z0, z1], integers, each low <= high")
            boxes.append(b)
            continue
        if op != "remove" and not isinstance(e["part"], dict):
            raise EditError(f"edit {n}: part must be an object")
        if op == "add":
            added.append(e["part"])
            continue
        k = e["index"]
        if type(k) is not int or not 0 <= k < len(local):
            raise EditError(f"edit {n}: index must be an integer in 0..{len(local) - 1}")
        i = local[k]
        if i in changes:
            raise EditError(f"edit {n}: more than one operation on input part {k}")
        changes[i] = e["part"] if op == "set" else None

    parts, sources, changed, removed = [], [], set(), set()
    for i, original in enumerate(body):
        p = changes.get(i, original)
        if p is None:
            removed.add(i)
            continue
        if i in changes:
            changed.add(len(parts))
        parts.append(deepcopy(p))
        sources.append(i)
    made: set = set()
    for b in boxes:  # after set/remove, before add: the cut is in the model the answer has kept
        before = {i: k for k, i in enumerate(sources) if i is not None}
        parts, sources, cut = _clear(parts, sources, b)
        removed |= {i for i in before if i not in set(sources)}
        changed = {k for k, i in enumerate(sources) if i is not None and i in changes} | cut
        made |= cut
    for p in added:
        changed.add(len(parts))
        parts.append(deepcopy(p))
        sources.append(None)
    if made:
        # a crumb of a cut part that ends up attached to nothing (a corner stud past the cut) is left out
        from .primitives import loose_parts

        # crumbs: small groups (6 cells or fewer) left attached to nothing once the cut is made; a bigger loose
        # group (a whole panel without its hinge) stays, for the checks to report
        from .kinematics import analyse

        loose = set(loose_parts(parts))
        bodies, joints = analyse(parts)
        link: dict = {}
        for j in joints:
            link.setdefault(j[0], set()).add(j[1])
            link.setdefault(j[1], set()).add(j[0])
        body_of = {k: b for b, m in bodies.items() for k in m}
        crumbs, seen = set(), set()
        for k in loose:
            if k in seen:
                continue
            group, todo = set(), [body_of[k]]
            while todo:
                b = todo.pop()
                if any(m in group for m in bodies[b]):
                    continue
                group |= set(bodies[b])
                todo += list(link.get(b, ()))
            seen |= group
            size = sum(parts[m].get("w", 1) * parts[m].get("l", 1) * H.get(parts[m]["type"], 1) for m in group)
            if size <= 6 and all(parts[m]["type"] in H for m in group):
                crumbs |= group
        if crumbs:
            keep = [k for k in range(len(parts)) if k not in crumbs]
            index = {k: n for n, k in enumerate(keep)}
            removed |= {sources[k] for k in crumbs if sources[k] is not None}
            changed = {index[k] for k in changed if k in index}
            parts, sources = [parts[k] for k in keep], [sources[k] for k in keep]
    return AppliedEdits(parts, sources, frozenset(changed), frozenset(removed))
