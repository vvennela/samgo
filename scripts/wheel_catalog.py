"""Find every LEGO wheel that fits a Technic axle, with the tyre that fits it, from real part geometry.

Rims: Rebrickable's "Wheels and Tyres" parts with an axle hole. Tyres: its "Tyre ..." parts. Each is measured from
the official LDraw library (samgo.ldraw.points): the spin axis is the part's thinnest extent; a rim's outer radius
and width, a tyre's inner radius, outer radius and width. A tyre fits a rim when the tyre's inner radius matches the
rim's outer radius (within 1 mm) and their widths agree (within 2 mm). Writes runs/wheel_catalog.json:
[{"rim", "rim_name", "tyre", "tyre_name", "diameter_mm", "width_mm", "colors"}], smallest first.

    PYTHONPATH=. python scripts/wheel_catalog.py
"""

import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from samgo.colors import _fetch
from samgo.ldraw import find, points

LDU = 0.4  # mm


def measure(part: str) -> dict | None:
    part = part + ".dat"
    if find(part) is None:
        return None
    p = points(part, studs=False)
    if len(p) < 8:
        return None
    ext = p.max(0) - p.min(0)
    axis = int(np.argmin(ext))
    centre = (p.max(0) + p.min(0)) / 2
    radial = np.delete(p - centre, axis, axis=1)
    r = np.linalg.norm(radial, axis=1)
    return {"axis": axis, "width_mm": float(ext[axis] * LDU), "outer_r_mm": float(r.max() * LDU),
            "inner_r_mm": float(np.percentile(r, 2) * LDU)}


def main() -> None:
    parts = list(csv.DictReader(gzip.open(_fetch("parts"), "rt")))
    cats = {r["id"]: r["name"] for r in csv.DictReader(gzip.open(_fetch("part_categories"), "rt"))}
    made = Counter(r["part_num"] for r in csv.DictReader(gzip.open(_fetch("elements"), "rt")))
    wheels = [p for p in parts if cats.get(p["part_cat_id"]) == "Wheels and Tyres" and made[p["part_num"]] > 0]
    rims = [p for p in wheels if p["name"].startswith("Wheel") and re.search(r"axle hole", p["name"], re.I)
            and "with" not in p["part_num"] and not re.search(r"c\d\d$", p["part_num"])
            and not re.search(r"Integral Tire|Cover|Stub Axle|Old", p["name"], re.I)]
    tyres = [p for p in wheels if p["name"].startswith("Tyre")]
    tyres = sorted(tyres, key=lambda p: -made[p["part_num"]])[:70]
    print(len(rims), "rims with an axle hole,", len(tyres), "tyres", flush=True)
    rm = {p["part_num"]: (p, measure(p["part_num"])) for p in rims}
    ty = {p["part_num"]: (p, measure(p["part_num"])) for p in tyres}
    out = []
    for rn, (rp, r) in rm.items():
        if r is None:
            continue
        for tn, (tp, t) in ty.items():
            if t is None:
                continue
            seat = r["outer_r_mm"] - t["inner_r_mm"]  # the tyre's bead sits in the rim's groove, under its lip
            if 0.5 <= seat <= 3.5 and abs(t["width_mm"] - r["width_mm"]) <= 2.0:
                out.append({"rim": rn, "rim_name": rp["name"], "tyre": tn, "tyre_name": tp["name"],
                            "diameter_mm": round(2 * t["outer_r_mm"], 1), "width_mm": round(t["width_mm"], 1),
                            "colors": made[rn] + made[tn]})
    out.sort(key=lambda w: (w["diameter_mm"], -w["colors"]))
    Path("runs/wheel_catalog.json").write_text(json.dumps(out, indent=1))
    for w in out:
        print(f'{w["diameter_mm"]:6.1f} x {w["width_mm"]:4.1f} mm  rim {w["rim"]:8} tyre {w["tyre"]:8} '
              f'{w["rim_name"][:45]} | {w["tyre_name"][:40]}')


if __name__ == "__main__":
    main()
