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


def apply_edits(body: list[dict], edits: list[dict], local: list[int] | None = None,
                allowed_ops: frozenset[str] = frozenset({"set", "remove", "add"})) -> AppliedEdits:
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
    changes, added = {}, []
    fields = {"set": {"op", "index", "part"}, "remove": {"op", "index"}, "add": {"op", "part"}}
    for n, e in enumerate(edits):
        if not isinstance(e, dict) or not isinstance(e.get("op"), str) or e["op"] not in fields:
            raise EditError(f"edit {n}: operation must be set, remove, or add")
        op = e["op"]
        if op not in allowed_ops:
            raise EditError(f"edit {n}: {op} is not allowed for this task; use {', '.join(sorted(allowed_ops))}")
        if set(e) != fields[op]:
            raise EditError(f"edit {n}: {op} requires exactly {', '.join(sorted(fields[op]))}")
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
    for p in added:
        changed.add(len(parts))
        parts.append(deepcopy(p))
        sources.append(None)
    return AppliedEdits(parts, sources, frozenset(changed), frozenset(removed))
