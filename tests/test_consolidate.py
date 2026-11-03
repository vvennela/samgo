"""samgo/consolidate.py builds exactly the pieces formal/MergeRule.lean proves correct (mergeX, mergeY, stack)."""
import random

from samgo.consolidate import VALID, _merged, consolidate


def cells(p):
    h = 1 if p["type"] == "plate" else 3
    return {(x, y, z) for x in range(p["x"], p["x"] + p["w"]) for y in range(p["y"], p["y"] + p["l"])
            for z in range(p["z"], p["z"] + h)}


def spec(p, q):
    """formal/MergeRule.lean: mergeX / mergeY preconditions and result (either order), else no merge."""
    for a, b in ((p, q), (q, p)):
        same = a["type"] == b["type"] and a["z"] == b["z"] and a["color"] == b["color"]
        if same and a["y"] == b["y"] and a["l"] == b["l"] and a["x"] + a["w"] == b["x"]:
            r = dict(a, w=a["w"] + b["w"])
            return r if tuple(sorted((r["w"], r["l"]))) in VALID else None
        if same and a["x"] == b["x"] and a["w"] == b["w"] and a["y"] + a["l"] == b["y"]:
            r = dict(a, l=a["l"] + b["l"])
            return r if tuple(sorted((r["w"], r["l"]))) in VALID else None
    return None


def test_merged_is_the_proved_rule_on_random_pairs():
    rng = random.Random(0)
    for _ in range(20000):
        def piece():
            return {"type": rng.choice(["plate", "brick"]), "x": rng.randint(0, 6), "y": rng.randint(0, 6),
                    "z": rng.randint(0, 2), "w": rng.randint(1, 4), "l": rng.randint(1, 4),
                    "color": rng.choice(["red", "blue"])}
        p, q = piece(), piece()
        r = _merged(p, q)
        assert r == spec(p, q)
        if r is not None:  # the proved theorems, checked on the instance: same cells, no overlap
            assert cells(r) == cells(p) | cells(q) and not cells(p) & cells(q)


def test_consolidate_keeps_every_cell_and_color():
    """An interlocked block: 1x6 plates along y, a layer of 1x1 plates (two colors), 6x1 plates along x on top.
    The 1x1 layer is one rigid body through the plates above and below, so it merges into bigger plates."""
    rng = random.Random(1)
    parts = [{"type": "plate", "x": x, "y": 0, "z": 0, "w": 1, "l": 6, "color": "light_gray"} for x in range(6)]
    parts += [{"type": "plate", "x": x, "y": y, "z": 1, "w": 1, "l": 1, "color": "red" if x < 3 else "light_gray"}
              for x in range(6) for y in range(6)]
    parts += [{"type": "plate", "x": 0, "y": y, "z": 2, "w": 6, "l": 1, "color": "light_gray"} for y in range(6)]
    rng.shuffle(parts)
    out, saved = consolidate(parts)
    colored = lambda ps: {c: p["color"] for p in ps for c in cells(p)}
    assert colored(out) == colored(parts) and len(out) == len(parts) - saved
    assert sum(p["z"] == 1 for p in out) <= 12  # 36 1x1 plates -> at most 12 real plates
