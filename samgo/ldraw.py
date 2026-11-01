"""Export SAMgo models to LDraw (.ldr) and check our part geometry against the official LDraw library.

Each part becomes the real LDraw part, placed by fitting the part's true bounding box (from the official library
geometry, stud bumps excluded) to the box our checkers assume: studded parts may only turn about the vertical,
axle-like parts may lie along any axis. What does not fit is reported per part: a real part larger or smaller than
our model of it, which is exactly the kind of error that would make a checked model unbuildable.

    uv run python -m samgo.ldraw runs/demo/chair/chair.json ...   # writes runs/ldraw/<name>.ldr and prints the check

LDraw frame: X = our x, Z = our y, -Y = up; 20 LDU a stud (8 mm), 8 LDU a plate (3.2 mm): 1 mm = 2.5 LDU.
"""

import itertools
import json
import re
import sys
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

LIB = Path(__file__).resolve().parent.parent / "runs" / "ldraw_lib"
BASE = "https://library.ldraw.org/library/official/"
UA = "SAMgo/0.1 (LDraw geometry check)"
STUD_PRIMS = re.compile(r"^(stud|stu2|stu3|stu4)", re.I)  # stud primitives (tops and underside tubes)


def _fetch(rel: str) -> Path | None:
    """runs/ldraw_lib/<rel>, downloaded from the official library on first use; None if it does not exist."""
    dest = LIB / rel
    if dest.exists():
        return dest
    miss = LIB / ".missing" / rel
    if miss.exists():
        return None
    for attempt in range(20):
        try:
            req = urllib.request.Request(BASE + rel, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:  # not in the library: remember, so the lookup order (parts/, p/) stays cheap
                miss.parent.mkdir(parents=True, exist_ok=True)
                miss.touch()
                return None
            if e.code != 429 or attempt == 19:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 5) + 0.5)  # the library allows 60 requests a minute
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def find(name: str) -> Path | None:
    """Resolve a sub-file reference the way LDraw does: parts/, then p/ (s\\... and 48\\... are subfolders)."""
    rel = name.strip().replace("\\", "/").lower()
    for root in ("parts/", "p/"):
        if (path := _fetch(root + rel)) is not None:
            return path
    return None


@lru_cache(maxsize=None)
def _parsed(name: str) -> tuple[str, list]:
    """(title, [(kind, data)]) of a library file: ("ref", (matrix 4x4, sub name)) or ("pts", array Nx3)."""
    path = find(name)
    if path is None:
        raise FileNotFoundError(f"{name} is not in the official LDraw library")
    title, items = "", []
    for k, line in enumerate(path.read_text(errors="replace").splitlines()):
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "0" and k == 0:
            title = line[1:].strip()
        elif tok[0] == "1" and len(tok) >= 15:
            v = [float(t) for t in tok[2:14]]
            m = np.eye(4)
            m[:3, 3] = v[0:3]
            m[:3, :3] = np.array(v[3:12]).reshape(3, 3)
            items.append(("ref", (m, " ".join(tok[14:]))))
        elif tok[0] in ("2", "3", "4", "5") and len(tok) >= 2 + 3 * int(tok[0] if tok[0] != "5" else 2):
            n = {"2": 2, "3": 3, "4": 4, "5": 2}[tok[0]]
            items.append(("pts", np.array([float(t) for t in tok[2:2 + 3 * n]]).reshape(n, 3)))
    return title, items


def title(name: str) -> str:
    return _parsed(name)[0]


@lru_cache(maxsize=None)
def _local_points(name: str, studs: bool) -> np.ndarray:
    """Every vertex of a library file in its own frame, sub-files resolved (without stud primitives if not studs)."""
    out = []
    for kind, data in _parsed(name)[1]:
        if kind == "pts":
            out.append(data)
            continue
        m, sub = data
        if not studs and STUD_PRIMS.match(Path(sub.replace("\\", "/")).name):
            continue
        pts = _local_points(sub, studs)
        if len(pts):
            out.append(pts @ m[:3, :3].T + m[:3, 3])
    return np.vstack(out) if out else np.zeros((0, 3))


def points(name: str, studs: bool = False) -> np.ndarray:
    return _local_points(name.lower(), studs)


def bbox(name: str, studs: bool = False) -> tuple[np.ndarray, np.ndarray]:
    p = points(name, studs)
    return p.min(axis=0), p.max(axis=0)


# ---- part numbers ----------------------------------------------------------------------------------------------

TECHNIC_BRICKS = {2: "3700", 4: "3701", 6: "3894", 8: "3702", 10: "2730", 12: "3895", 16: "3703"}
AXLES = {2: "32062", 3: "4519", 4: "3705", 5: "32073", 6: "3706", 7: "44294", 8: "3707", 9: "60485", 10: "3737",
         11: "23948", 12: "3708", 16: "50451", 32: "50450"}
STOP_AXLES = {3: "24316", 4: "87083", 5: "15462", 8: "55013"}
TILES = {(1, 1): "3070b", (1, 2): "3069b", (1, 3): "63864", (1, 4): "2431", (1, 6): "6636", (1, 8): "4162",
         (2, 2): "3068b"}
SIMPLE = {"bush": "3713", "half_bush": "32123b", "axle_pin": "43093", "axle_joiner": "6538a", "cross_block": "6536",
          "hinge_base": "2429", "hinge_top": "2430", "click_a": "44301a", "click_b": "44302a",
          "turntable_base": "3680", "turntable_top": "3679", "car_door": "3821", "bar_plate": "60478",
          "clip_plate": "63868"}
EXTRA_COLORS = {"trans_clear": 47, "trans_black": 40}
DEFAULT_COLOR = {"axle": 0, "pin": 0, "bush": 71, "half_bush": 71, "axle_pin": 1, "liftarm": 71}
STUDDED = {"plate", "brick", "technic_brick", "tile", "hinge_base", "hinge_top", "click_a", "click_b", "bar_plate", "clip_plate",
           "turntable_base", "turntable_top", "car_door"}
AXIS_PARTS = {"axle", "pin", "bush", "half_bush", "axle_pin", "axle_joiner"}


def part_ids(p: dict) -> list[str]:
    """The LDraw part(s) for one SAMgo part (a wheel is a tyre and a rim)."""
    from .parts import LDRAW_IDS
    from .technic import TECHNIC_LDRAW, WHEELS

    t = p["type"]
    if t in ("plate", "brick"):
        return [LDRAW_IDS[(t, *sorted((p["w"], p["l"])))]]
    if t == "technic_brick":
        return [TECHNIC_BRICKS[max(p["w"], p["l"])]]
    if t == "tile":
        return [TILES[tuple(sorted((p["w"], p["l"])))]]
    if t == "axle":
        return [(STOP_AXLES if p.get("stop") else AXLES)[int(p["length"])]]
    if t == "wheel":
        _, _, tyre, rim = WHEELS[p.get("size", "30.4x14")]
        return [tyre, rim]
    if t == "pin":
        return [TECHNIC_LDRAW[("pin", p.get("length", 2), p.get("friction", True))]]
    if t == "liftarm":
        return [TECHNIC_LDRAW[("liftarm", p["length"])]]
    return [SIMPLE[t]]


def color_code(p: dict, part_num: str | None = None) -> int:
    """The LDraw color for `p`; with `part_num`, one that part was actually produced in (colors.legal_code)."""
    from .parts import COLORS

    c = p.get("color")
    code = COLORS[c][0] if c in COLORS else EXTRA_COLORS.get(c, DEFAULT_COLOR.get(p["type"], 71))
    if part_num is None:
        return code
    from .colors import legal_code

    return legal_code(p | {"color": c if c in COLORS or c in EXTRA_COLORS else None}, part_num, [code])


# ---- geometry ---------------------------------------------------------------------------------------------------

def _rotations(vertical_only: bool) -> list[np.ndarray]:
    if vertical_only:  # about LDraw Y (up): studs stay on top
        return [np.array(m, float) for m in ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                                             [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]])]
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, sg) in enumerate(zip(perm, signs)):
                m[row, col] = sg
            if np.linalg.det(m) > 0:
                out.append(m)
    out.sort(key=lambda m: -np.trace(m))  # identity first, so ties keep the part as modelled
    return out


def expected_box(p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(low, high, compare) in mm in our frame: the volume our checkers assume, and which axes have a known size
    (for axle-like parts only the length is ours to claim; their cross-section is the real part's)."""
    from .kinematics import AXES, _start, box
    from .technic import LENGTH, STUD, WHEELS, wheel_studs

    t = p["type"]
    if t in AXIS_PARTS:
        along = AXES[p.get("axis", "y")]
        n = {"axle": p.get("length", 2), "pin": p.get("length", 2), "bush": 1}.get(t, LENGTH.get(t, 1))
        a = _start(p)
        b = a + n * STUD * along
        r = 4.0 * (1 - along)
        return np.minimum(a, b) - r, np.maximum(a, b) + r, along.astype(bool)
    if t == "wheel":
        size = p.get("size", "30.4x14")
        d, width = WHEELS[size][0], WHEELS[size][1]
        along = AXES[p.get("axis", "y")]
        c = _start(p) + wheel_studs(size) * STUD / 2 * along
        half = d / 2 * (1 - along) + width / 2 * along
        return c - half, c + half, np.ones(3, bool)
    lo, hi = box(p)
    return np.asarray(lo, float), np.asarray(hi, float), np.ones(3, bool)


AXES_MM = {"x": np.array([1.0, 0, 0]), "y": np.array([0.0, 1, 0]), "z": np.array([0.0, 0, 1])}


def _to_ldraw(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], -v[2], v[1]]) * 2.5


def _place_swivel(p: dict, name: str) -> tuple[str, float]:
    """2429/2430: the library origin is the pivot, on top of the plate; the base reaches -X from it, the top +X. Each
    real half is 2.4 studs long because their round knuckles overlap at the pivot; together they span the same 1x4
    our two 1x2 halves do, so they are placed by the pivot, not by box fit."""
    px, py = p["pin_x"] * 20, (p["y"] + p.get("l", 1) / 2) * 20
    if p.get("l", 1) > p.get("w", 2):  # a half running along our y
        px, py = (p["x"] + p["w"] / 2) * 20, p["pin_y"] * 20
    centre = np.array([(p["x"] + p["w"] / 2) * 20, (p["y"] + p["l"] / 2) * 20])
    away = centre - np.array([px, py])                       # pivot -> this half, in LDraw X/Z
    local = np.array([-1.0, 0, 0]) if p["type"] == "hinge_base" else np.array([1.0, 0, 0])
    R = max(_rotations(True), key=lambda m: (m @ local)[[0, 2]] @ away)
    top = -(p["z"] + 1) * 8
    m = " ".join(f"{v:g}" for v in R.ravel())
    return f"1 {color_code(p, name)} {px:g} {top:g} {py:g} {m} {name}.dat", 0.0


def _place_clip(p: dict, name: str) -> tuple[str, float]:
    """60478 / 63868: the library plate is centred at its origin, top at Y=0, its bar / clip end toward +X.
    Turned so that end faces `facing`; the plate body sits exactly on our footprint."""
    turn = {"+x": "1 0 0 0 1 0 0 0 1", "-x": "-1 0 0 0 1 0 0 0 -1", "+y": "0 0 -1 0 1 0 1 0 0",
            "-y": "0 0 1 0 1 0 -1 0 0"}[p.get("facing", "+x")]
    cx, cz = (p["x"] + p["w"] / 2) * 20, (p["y"] + p["l"] / 2) * 20
    return f"1 {color_code(p, name)} {cx:g} {-(p['z'] + 1) * 8:g} {cz:g} {turn} {name}.dat", 0.0


def place(p: dict, name: str) -> tuple[str, float]:
    """One LDraw line for library part `name` standing in for SAMgo part `p`, and the worst size mismatch (mm)."""
    if p["type"] in ("bar_plate", "clip_plate"):
        return _place_clip(p, name)
    if p["type"] in ("hinge_base", "hinge_top") and "pin_x" in p:
        return _place_swivel(p, name)
    lo, hi, compare = expected_box(p)
    want_c = _to_ldraw((lo + hi) / 2)
    want = np.abs(_to_ldraw(hi - lo))
    cmp_l = np.abs(_to_ldraw(compare.astype(float))) > 0
    llo, lhi = bbox(name + ".dat")
    size, centre = lhi - llo, (llo + lhi) / 2
    vertical_only = p["type"] in STUDDED
    if p["type"] == "wheel" and name != part_ids(p)[0]:  # the rim: concentric with the tyre, its own size
        cmp_l = np.zeros(3, bool)
    axial = p["type"] in AXIS_PARTS or p["type"] == "wheel"
    if axial:  # the real part's round axis: the one dimension unlike the other two
        odd = int(np.argmax([abs(size[i] - np.mean(np.delete(size, i))) for i in range(3)]))
        ours = int(np.argmax(np.abs(_to_ldraw(AXES_MM[p.get("axis", "y")]))))
    best = None
    for R in _rotations(vertical_only):
        if axial and int(np.argmax(np.abs(R[:, odd]))) != ours:
            continue  # it must turn about our axis line
        got = np.abs(R @ size)
        err = np.where(cmp_l, np.abs(got - want), 0).max() if cmp_l.any() else 0.0
        if best is None or err < best[0] - 1e-9:
            best = (err, R)
    err, R = best
    t = want_c - R @ centre
    m = " ".join(f"{v:g}" for v in R.ravel())
    return f"1 {color_code(p, name)} {t[0]:g} {t[1]:g} {t[2]:g} {m} {name}.dat", err / 2.5


def export(parts: list[dict], name: str = "model") -> tuple[str, list[dict]]:
    """The .ldr text (one step per layer, bottom up) and the per-part geometry report."""
    lines = [f"0 {name}", f"0 Name: {name}.ldr", "0 Author: SAMgo", "0 !LDRAW_ORG Unofficial_Model"]
    report, z = [], None
    for k in sorted(range(len(parts)), key=lambda i: parts[i].get("z", 0)):
        p = parts[k]
        if z is not None and p.get("z", 0) != z:
            lines.append("0 STEP")
        z = p.get("z", 0)
        for pid in part_ids(p):
            line, err = place(p, pid)
            lines.append(line)
            report.append({"index": k, "type": p["type"], "ldraw": pid, "title": title(pid + ".dat"),
                           "mismatch_mm": round(err, 2)})
    lines.append("0 STEP")
    return "\n".join(lines) + "\n", report


def main() -> None:
    out_dir = LIB.parent / "ldraw"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in map(Path, sys.argv[1:]):
        parts = json.loads(path.read_text())
        name = f"{path.parent.name}_{path.stem}"
        text, report = export(parts, name)
        (out_dir / f"{name}.ldr").write_text(text)
        bad = [r for r in report if r["mismatch_mm"] > 1.0]
        worst = max((r["mismatch_mm"] for r in report), default=0)
        print(f"{name}: {len(parts)} parts -> {len(report)} LDraw parts; worst size mismatch {worst:.2f} mm; "
              f"{len(bad)} over 1 mm")
        seen = set()
        for r in sorted(bad, key=lambda r: -r["mismatch_mm"]):
            if (r["type"], r["ldraw"]) not in seen:
                seen.add((r["type"], r["ldraw"]))
                print(f"   {r['type']} -> {r['ldraw']} ({r['title']}): {r['mismatch_mm']} mm")


if __name__ == "__main__":
    main()
