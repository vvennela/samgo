"""Colors that exist: every part in a color LEGO actually made, mechanisms in standard Technic colors.

The catalogue is Rebrickable's element list (part number x color id; its color ids follow LDraw's), downloaded
once into runs/rebrickable/. A part whose color was never produced takes the nearest produced color from the
model's own palette, so a model never asks for an impossible element and the color scheme stays coherent.

    from samgo.colors import legalize
    parts, changes = legalize(parts)
"""

import csv
import gzip
from functools import lru_cache
from pathlib import Path

from .parts import COLORS

ROOT = Path(__file__).resolve().parent.parent / "runs" / "rebrickable"
URL = "https://cdn.rebrickable.com/media/downloads/{}.csv.gz"
NAMED = {name: code for name, (code, _) in COLORS.items()} | {"trans_clear": 47, "trans_black": 40}
CODE_NAME = {code: name for name, code in NAMED.items()}
TRANS = {47, 40}
# Mechanism parts in the colors real Technic sets use them, first choice first (LDraw codes).
STANDARD = {"tyre": (0,), "rim": (71, 15, 72, 0), "axle": (0, 71, 72), "bush": (71, 72, 0), "half_bush": (71, 72, 0),
            "pin": (0, 71, 72, 1), "liftarm": (71, 0, 72), "turntable": (0, 71, 72)}


def _fetch(name: str) -> Path:
    path = ROOT / f"{name}.csv.gz"
    if not path.exists():
        import urllib.request

        ROOT.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL.format(name), headers={"User-Agent": "SAMgo/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    return path


@lru_cache(maxsize=1)
def catalogue() -> tuple[dict[str, set[int]], dict[int, tuple[int, int, int]]]:
    """(part number -> color codes it was produced in, color code -> RGB)."""
    made: dict[str, set[int]] = {}
    for r in csv.DictReader(gzip.open(_fetch("elements"), "rt")):
        made.setdefault(r["part_num"], set()).add(int(r["color_id"]))
    rgb = {int(r["id"]): tuple(int(r["rgb"][i:i + 2], 16) for i in (0, 2, 4))
           for r in csv.DictReader(gzip.open(_fetch("colors"), "rt"))}
    return made, rgb


def colors_of(part_num: str) -> set[int] | None:
    """Codes a part was made in; None if the catalogue does not know the part (old/renamed numbers tried too)."""
    made, _ = catalogue()
    base = part_num.rstrip("abcdefghij")
    for key in (part_num, base, base + "a", base + "b", base + "c"):
        if key in made:
            return made[key]
    return None


def exists(part_num: str, code: int) -> bool:
    got = colors_of(part_num)
    return got is None or code in got  # unknown parts are not second-guessed


def _distance(a: int, b: int) -> float:
    _, rgb = catalogue()
    if a not in rgb or b not in rgb:
        return 1e9
    return sum((x - y) ** 2 for x, y in zip(rgb[a], rgb[b])) + (0 if (a in TRANS) == (b in TRANS) else 1e7)


def role(p: dict, part_num: str) -> str | None:
    """Which standard-color rule a functional part follows (a wheel is a tyre and a rim)."""
    from .technic import WHEELS

    t = p["type"]
    if t == "wheel":
        _, _, tyre, rim = WHEELS[p.get("size", "30.4x14")]
        return "tyre" if part_num == tyre else "rim"
    if t.startswith("turntable"):
        return "turntable"
    return t if t in STANDARD else None


def legal_code(p: dict, part_num: str, palette: list[int]) -> int:
    """The color code to use for library part `part_num` standing in for `p`."""
    want = NAMED.get(p.get("color"))
    r = role(p, part_num)
    if r is not None:
        for code in ((want,) if want is not None and r not in ("tyre",) else ()) + STANDARD[r]:
            if code is not None and exists(part_num, code):
                return code
    if want is not None and exists(part_num, want):
        return want
    made = colors_of(part_num) or set()
    named = [c for c in made if c in CODE_NAME]
    pool = [c for c in palette if c in named] or named or [want if want is not None else 71]
    target = want if want is not None else 71
    return min(pool, key=lambda c: _distance(c, target))


def split_to_produced(p: dict) -> list[dict] | None:
    """A 1-wide or 2-wide plate/brick cut along its length into the longest pieces produced in its color."""
    from .parts import LDRAW_IDS

    code = NAMED[p["color"]]
    along_x = p["w"] >= p["l"]
    length, width = (p["w"], p["l"]) if along_x else (p["l"], p["w"])
    sizes = [n for n in (8, 6, 4, 3, 2, 1) if (p["type"], *sorted((width, n))) in LDRAW_IDS
             and exists(LDRAW_IDS[(p["type"], *sorted((width, n)))], code)]
    if not sizes:
        return None
    pieces, at = [], 0
    while at < length:
        n = next((n for n in sizes if n <= length - at), None)
        if n is None:
            return None
        q = p | ({"w": n, "x": p["x"] + at} if along_x else {"l": n, "y": p["y"] + at})
        pieces.append(q)
        at += n
    return pieces


def legalize(parts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Parts recolored to produced colors (functional parts to standard Technic colors where theirs is not made),
    and the list of changes. Wheels keep one color field (the rim's); the tyre is always exported black."""
    from .ldraw import part_ids

    palette = sorted({NAMED[p["color"]] for p in parts if p.get("color") in NAMED},
                     key=lambda c: -sum(NAMED.get(p.get("color")) == c for p in parts))
    out, changes = [], []
    for i, p in enumerate(parts):
        if p["type"] in ("plate", "brick") and NAMED.get(p.get("color")) in TRANS \
                and not exists(part_ids(p)[0], NAMED[p["color"]]):
            pieces = split_to_produced(p)
            if pieces:  # glass stays glass: the same cells in smaller pieces made in that color
                changes.append({"index": i, "type": p["type"], "part": part_ids(p)[0], "from": p["color"],
                                "to": f"{len(pieces)} pieces of {p['color']}"})
                out += pieces
                continue
        pid = part_ids(p)[-1]  # a wheel's color field is its rim's
        code = legal_code(p, pid, palette)
        name = CODE_NAME.get(code, p.get("color"))
        if name != p.get("color"):
            changes.append({"index": i, "type": p["type"], "part": pid, "from": p.get("color"), "to": name})
            p = p | {"color": name}
        out.append(p)
    return out, changes


def paint(parts: list[dict], body: str, trim: str = "black") -> list[dict]:
    """A clean color scheme for a car (color is paint: it is not checked and does not constrain the layout): the
    body in one color; the Technic frame, the layer it hangs from (floor) and everything below it in `trim`; glass
    and mechanism parts keep theirs."""
    frame = [p for p in parts if p["type"] == "technic_brick"]
    floor = (min(p["z"] for p in frame) + 3) if frame else None
    out = []
    for p in parts:
        if p["type"] in ("plate", "brick") and p.get("color") not in ("trans_clear", "trans_black"):
            low = floor is not None and p["z"] <= floor
            p = p | {"color": trim if low else body}
        elif p["type"] == "technic_brick":
            p = p | {"color": trim}
        out.append(p)
    return out


def body_color(target) -> str:
    """The most common color of a voxel target (the photo's main paint color)."""
    import numpy as np

    from .parts import COLORS

    names = list(COLORS)
    cells = np.asarray(target)[np.asarray(target) > 0]
    return names[int(np.bincount(cells).argmax()) - 1] if cells.size else "light_gray"
