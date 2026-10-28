"""Turn a failed verifier report into corrections the model can act on.

The verifier says what is wrong (codes plus piece indices); this names the model's own
pieces, the cells that are missing or extra, and the rule each one breaks, so a retry
can fix the build instead of guessing. Indices are over given + added; only the added
pieces are the model's, and they are numbered from 1 in its own answer order.
"""

import json

import numpy as np

from samgo.parts import COLOR_INDEX, PartError, parse_piece
from samgo.verifier import Report

MAX_ITEMS = 12  # per list, so a badly broken build doesn't produce a wall of text
NAMES = {i: name for name, i in COLOR_INDEX.items()}


def _piece(raw) -> str:
    """'2x4 brick at (x,y,z)'; anything unreadable as its raw text."""
    try:
        return f"{raw['w']}x{raw['l']} {raw['type']} at ({raw['x']},{raw['y']},{raw['z']})"
    except (TypeError, KeyError):
        return (json.dumps(raw) if isinstance(raw, dict) else str(raw))[:120]


def _cap(items: list[str]) -> list[str]:
    return items[:MAX_ITEMS] + ([f"... and {len(items) - MAX_ITEMS} more"] if len(items) > MAX_ITEMS else [])


def _own(indices: list[int], given: list, added: list, note=None) -> list[str]:
    """The model's pieces among `indices`, as '#n 2x4 brick at (x,y,z)' plus an optional note."""
    return _cap([f"#{i - len(given) + 1} {_piece(added[i - len(given)])}" + (note(i) if note else "")
                 for i in indices if i >= len(given)])


def _cells(mask: np.ndarray, label=None) -> str:
    """Cells of `mask` grouped by layer: 'z=3: (1,2) (1,3)'."""
    xs, ys, zs = np.nonzero(mask)
    by_z: dict[int, list[str]] = {}
    for x, y, z in zip(xs, ys, zs):
        by_z.setdefault(int(z), []).append(f"({x},{y})" + (f" {label(x, y, z)}" if label else ""))
    parts, shown = [], 0
    for z in sorted(by_z):
        cells = by_z[z][: max(MAX_ITEMS - shown, 0)]
        if cells:
            parts.append(f"z={z}: " + " ".join(cells))
            shown += len(cells)
    total = len(xs)
    return "; ".join(parts) + (f"; ... {total - shown} more" if total > shown else "")


def _occupancy(pieces: list, shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    built = np.zeros(shape, dtype=bool)
    colors = np.zeros(shape, dtype=np.int32)
    for raw in pieces:
        try:
            p = parse_piece(raw)
        except PartError:
            continue
        cells = tuple(slice(max(s.start, 0), max(min(s.stop, n), 0)) for s, n in zip(p.cells, shape))
        built[cells] = True
        colors[cells] = COLOR_INDEX[p.color]
    return built, colors


def describe(report: Report, given: list, added: list, target: np.ndarray) -> str:
    """Feedback for one failed attempt; empty if the report passed."""
    if report.passed:
        return ""
    if not added:
        return "- Your answer had no pieces. Output one JSON object per piece, one per line."
    lines = []
    for issue in report.issues:
        if issue.code == "invalid_part":
            continue  # grouped below: one line per distinct reason
        elif issue.code == "out_of_bounds":
            lines.append(f"- Outside the target grid {target.shape}: " + ", ".join(_own(issue.pieces, given, added)))
        elif issue.code == "overlap":
            lines.append("- These pieces fill cells that another piece already fills: "
                         + ", ".join(_own(issue.pieces, given, added)))
        elif issue.code == "floating":
            pieces = given + added
            lines.append("- Not supported (a piece connects only to one directly below or above that covers a shared "
                         "(x, y) stud, or it must sit at z=0): "
                         + ", ".join(_own(issue.pieces, given, added, lambda i: _below(pieces, i, target.shape))))
        elif issue.code == "disconnected":
            lines.append("- These stand on the ground but are not joined to the main build through stacked pieces: "
                         + ", ".join(_own(issue.pieces, given, added)))
        elif issue.code == "off_target":
            built, colors = _occupancy(given + added, target.shape)
            wanted = target > 0
            if (missing := wanted & ~built).any():
                lines.append(f"- Target cells left empty: {_cells(missing)}")
            if (extra := built & ~wanted).any():
                lines.append(f"- Cells filled that must stay empty: {_cells(extra)}")
            wrong = built & wanted & (colors != target)
            if wrong.any():
                lines.append("- Wrong color: " + _cells(wrong, lambda x, y, z: f"needs {NAMES[int(target[x, y, z])]}"))
        else:
            lines.append(f"- {issue.message}")
    reasons: dict[str, list[int]] = {}
    for issue in report.issues:
        if issue.code == "invalid_part":
            reasons.setdefault(issue.message.split(": ", 1)[-1], []).extend(issue.pieces)
    for reason, indices in reasons.items():
        lines.insert(0, f"- Not a valid part ({reason}): {', '.join(_own(indices, given, added))}. "
                        "Use only the listed footprints, type plate or brick, and the listed colors.")
    return "\n".join(lines)


def _below(pieces: list, i: int, shape: tuple) -> str:
    """Where the build under piece i actually ends, so a wrong z is visible."""
    try:
        p = parse_piece(pieces[i])
    except PartError:
        return ""
    built, _ = _occupancy(pieces[:i] + pieces[i + 1:], shape)
    column = built[p.cells[0], p.cells[1], : max(min(p.z, shape[2]), 0)]
    filled = np.nonzero(column.any(axis=(0, 1)))[0]
    if not len(filled):
        return " [nothing under it]"
    top = int(filled.max()) + 1
    return f" [build under it tops out at z={top}]" if top != p.z else ""


def retry_message(feedback: str) -> str:
    return ("The checker rejected that build:\n" + feedback
            + "\n\nFix these problems. Output the complete corrected list of pieces you add "
              "(not the already-built ones), one JSON object per line and nothing else.")
