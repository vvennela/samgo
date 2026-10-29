import pytest

from samgo.palette import KINDS, PaletteError, Part, check, count, ldr, parse, stability

SHAPE = (12, 12, 30)


def P(line):
    return parse(line)


def test_palette_has_35_parts_of_9_kinds():
    assert count() == 35 and len(KINDS) == 9


def test_lines_round_trip_and_bad_lines_are_rejected():
    for line in ("brick 2x4 (0,0,0)", "plate 4x2 (1,1,3)", "slope 2x2 (0,0,3) -y", "slope 2x1 (0,0,3) +x",
                 "curve 1x2 (3,3,6) +y", "arch 1x4 (0,0,3)"):
        assert P(line).line() == line
    for bad in ("brick 3x3 (0,0,0)", "slope 2x2 (0,0,0)", "slope 2x2 (0,0,0) +z", "slope 2x3 (0,0,0) +y",
                "tile 1x1 (0,0,0)", "plate 2x2 (0,0,0) +x", "gear 2x2 (0,0,0)", "brick 2x2 (0,0)"):
        with pytest.raises(PaletteError):
            P(bad)


def test_slope_has_studs_only_on_its_high_row():
    s = P("slope 2x2 (0,0,0) -y")  # goes down toward -y, so the high row is y = 1
    assert s.studs == {(0, 1), (1, 1)} and s.grips == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert P("slope 2x1 (0,0,0) +x").studs == {(0, 0)}


def test_a_plate_on_the_low_side_of_a_slope_floats_but_on_the_high_side_holds():
    base = [P("brick 2x2 (0,0,0)"), P("slope 2x2 (0,0,3) -y")]
    assert not check(base + [P("plate 2x1 (0,0,6)")], SHAPE)["ok"]
    assert check(base + [P("plate 2x1 (0,1,6)")], SHAPE)["ok"]


def test_nothing_attaches_on_top_of_a_tile():
    r = check([P("brick 2x2 (0,0,0)"), P("tile 2x2 (0,0,3)"), P("brick 2x2 (0,0,4)")], SHAPE)
    assert ("floating", [2]) in r["issues"]


def test_an_arch_stands_on_its_ends_and_bridges_the_gap():
    legs = [P("brick 1x1 (0,0,0)"), P("brick 1x1 (3,0,0)")]
    r = check(legs + [P("arch 4x1 (0,0,3)")], SHAPE)
    assert r["ok"] and r["components"] == 1
    # A post under the middle of the arch is not attached to it.
    assert not check([P("brick 1x1 (1,0,0)"), P("arch 4x1 (0,0,3)")], SHAPE)["ok"]


def test_stability_of_a_simple_stack_and_a_one_stud_overhang():
    assert stability([P("brick 2x4 (0,0,0)"), P("plate 2x4 (0,0,3)")])["ok"]
    hang = [P("brick 1x6 (0,0,0)"), P("plate 1x6 (5,0,3)")]
    assert stability(hang, clutch=5)["ok"] is False


def test_ldraw_uses_checked_part_ids_and_turns_sloped_parts():
    assert P("plate 1x2 (0,0,0)").ldr().endswith("3023b.dat")
    a, b = P("slope 2x2 (0,0,3) -y").ldr(), P("slope 2x2 (0,0,3) +y").ldr()
    assert a.endswith("3039.dat") and " 1 0 0 0 1 0 0 0 1 " in a and " -1 0 0 0 1 0 0 0 -1 " in b
    assert P("slope 1x2 (0,0,0) -y").ldr().endswith("3040b.dat") and P("slope 4x2 (0,0,0) -y").ldr().endswith("3037.dat")
    assert ldr([P("brick 2x2 (0,0,0)")]).count("\n1 ") == 1
