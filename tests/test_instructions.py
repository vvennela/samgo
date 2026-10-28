from samgo.instructions import name, steps
from tests.test_technic import car


def test_every_part_appears_once_bottom_up_with_axles_then_wheels_last():
    parts = car()
    plan = steps(parts)
    flat = [p for step in plan for p in step]
    assert len(flat) == len(parts) and all(any(p is q for q in flat) for p in parts)
    solid_z = [p["z"] for step in plan[:-2] for p in step]
    assert solid_z == sorted(solid_z)
    assert {p["type"] for p in plan[-2]} == {"axle"} and {p["type"] for p in plan[-1]} == {"wheel"}


def test_part_names_read_like_a_parts_list():
    assert name({"type": "plate", "w": 8, "l": 2, "color": "light_gray"}) == "2x8 plate, light gray"
    assert name({"type": "axle", "length": 6}) == "Technic axle 6"
