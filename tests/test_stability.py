from samgo.stability import stability
from tests.test_technic import car


def piece(w, l, type, x, y, z, color="red"):
    return {"w": w, "l": l, "type": type, "x": x, "y": y, "z": z, "color": color}


def test_tower_stands_on_compression_alone():
    report = stability([piece(1, 1, "brick", 0, 0, 3 * k) for k in range(10)])
    assert report.passed and report.metrics["peak_tension"] < 1e-6


def test_long_plate_hanging_from_one_stud_is_overloaded():
    # The base is long enough that nothing tips; only the stud joint is under strain.
    report = stability([piece(8, 1, "brick", 0, 0, 0), piece(8, 1, "plate", 7, 0, 3)], clutch=25)
    assert not report.passed and report.codes == {"overloaded"}


def test_same_plate_held_by_two_studs_stands():
    assert stability([piece(8, 1, "brick", 0, 0, 0), piece(8, 1, "plate", 6, 0, 3)], clutch=25).passed


def test_heavy_load_on_a_one_stud_joint_breaks_it_at_real_clutch():
    counterweight = [piece(2, 2, "brick", 0, 0, 3 + 3 * k) for k in range(12)]  # keeps the whole from tipping
    load = [piece(2, 2, "brick", 13, 0, 4 + 3 * k) for k in range(6)]          # out on the arm
    report = stability([piece(8, 2, "brick", 0, 0, 0), piece(8, 2, "plate", 7, 0, 3), *counterweight, *load])
    assert report.codes == {"overloaded"}


def test_a_top_heavy_build_tips_over():
    assert stability([piece(1, 1, "brick", 0, 0, 0), piece(8, 1, "plate", 0, 0, 3)]).codes == {"unbalanced"}


def test_balanced_over_its_support_needs_no_pull():
    report = stability([piece(1, 1, "brick", 2, 0, 0), piece(5, 1, "plate", 0, 0, 3)])
    assert report.passed and report.metrics["peak_tension"] < 1e-6


def test_a_floating_piece_cannot_balance():
    assert stability([piece(2, 2, "brick", 0, 0, 0), piece(2, 2, "brick", 5, 5, 3)]).codes == {"unbalanced"}


def test_the_reference_car_stands_on_its_axles():
    assert stability(car(), vehicle=True).passed
