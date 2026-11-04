"""Which plain plates and bricks LEGO currently makes, by colour: a part counts as current in a colour if it was in
at least MIN_SETS sets released since SINCE (Rebrickable's public set inventories).

    python scripts/lego_current.py <dir with Rebrickable parts/sets/inventories/inventory_parts/colors .csv>
writes samgo/current_parts.json: {"since":..., "parts": {"plate 2x16": {"part": "4282", "colors": {"red": n_sets}}}}
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SINCE, MIN_SETS = 2021, 3
OURS = {"Black": "black", "Blue": "blue", "Green": "green", "Red": "red", "Yellow": "yellow", "White": "white",
        "Orange": "orange", "Tan": "tan", "Reddish Brown": "reddish_brown", "Light Bluish Gray": "light_gray",
        "Dark Bluish Gray": "dark_gray", "Dark Blue": "dark_blue", "Trans-Clear": "trans_clear",
        "Trans-Black": "trans_black"}
NAME = re.compile(r"^(Plate|Brick) (\d+) x (\d+)$")


def rows(d, f):
    return csv.DictReader(open(Path(d) / f"{f}.csv", newline=""))


def main():
    d = sys.argv[1]
    plain = {}
    for r in rows(d, "parts"):
        m = NAME.match(r["name"])
        if m:
            a, b = sorted((int(m[2]), int(m[3])))
            plain[r["part_num"]] = f"{m[1].lower()} {a}x{b}"
    colors = {r["id"]: OURS[r["name"]] for r in rows(d, "colors") if r["name"] in OURS}
    year = {r["set_num"]: int(r["year"]) for r in rows(d, "sets")}
    inv = {r["id"]: r["set_num"] for r in rows(d, "inventories") if year.get(r["set_num"], 0) >= SINCE}
    sets = defaultdict(set)
    for r in rows(d, "inventory_parts"):
        if r["inventory_id"] in inv and r["part_num"] in plain and r["color_id"] in colors:
            sets[(plain[r["part_num"]], r["part_num"], colors[r["color_id"]])].add(inv[r["inventory_id"]])
    out = {}
    for (kind, num, col), s in sets.items():
        if len(s) < MIN_SETS:
            continue
        e = out.setdefault(kind, {"part": num, "colors": {}, "_n": 0})
        if num != e["part"]:          # several moulds of one size: keep the one in the most sets
            if len(s) <= e["_n"]:
                continue
            e.update(part=num, colors={}, _n=0)
        e["colors"][col] = len(s)
        e["_n"] = max(e["_n"], len(s))
    for e in out.values():
        e.pop("_n")
    path = Path(__file__).resolve().parents[1] / "samgo" / "current_parts.json"
    path.write_text(json.dumps({"since": SINCE, "min_sets": MIN_SETS, "source": "rebrickable.com downloads",
                                "parts": dict(sorted(out.items()))}, indent=1))
    for k, e in sorted(out.items()):
        print(k, e["part"], len(e["colors"]), "colours")


if __name__ == "__main__":
    main()
