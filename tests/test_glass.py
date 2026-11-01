from samgo.glass import GLASS, add_glass


def car_like():
    """A 10x6 body two bricks tall with a 6x6 cabin of 1x1 bricks on top, and a roof plate."""
    parts = [{"type": "brick", "w": 10, "l": 6, "x": 0, "y": 0, "z": z, "color": "red"} for z in (0, 3)]
    for x in range(2, 8):
        for y in range(6):
            parts.append({"type": "brick", "w": 1, "l": 1, "x": x, "y": y, "z": 6, "color": "red"})
    parts.append({"type": "plate", "w": 6, "l": 6, "x": 2, "y": 0, "z": 9, "color": "red"})
    return parts


def test_cabin_walls_become_glass_but_pillars_roof_and_body_do_not():
    out, n = add_glass(car_like())
    glass = {(p["x"], p["y"]) for p in out if p["color"] == GLASS}
    assert n == len(glass) > 0
    assert (3, 0) in glass and (2, 2) in glass            # a side window and the windscreen
    assert (2, 0) not in glass and (7, 5) not in glass    # corner pillars (A/C)
    assert (4, 0) not in glass and (4, 5) not in glass    # middle pillar (B)
    assert (4, 2) not in glass                            # the cabin's inside
    assert all(p["color"] == "red" for p in out if p["z"] in (0, 3, 9))   # body and roof stay solid


def test_no_cabin_no_glass():
    box = [{"type": "brick", "w": 4, "l": 4, "x": 0, "y": 0, "z": z, "color": "red"} for z in (0, 3, 6)]
    assert add_glass(box) == (box, 0)
