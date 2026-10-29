import numpy as np

from brickgpt.decode import Config, generate
from brickgpt.shape import ShapeChecked, fit_target
from brickgpt.structure import Structure, parse
from tests.test_brickgpt_decode import Scripted

RED = 4


def box(x=4, y=4, z=2):
    t = np.zeros((20, 20, 20), dtype=np.int32)
    t[:x, :y, :z] = RED
    return t


def checked(target, **kw):
    return ShapeChecked(Structure(), target, parse, **kw)


def test_fit_target_turns_plates_into_brick_layers_by_majority():
    plates = np.zeros((4, 4, 6), dtype=np.int32)
    plates[:, :, 0:3] = RED          # a full brick layer
    plates[0, 0, 3] = RED            # 1 of 3 plates: not enough for the second layer
    plates[1, 1, 3:5] = RED          # 2 of 3 plates: counts
    t = fit_target(plates, (20, 20, 20), 3)
    assert t[:4, :4, 0].all() and t[1, 1, 1] == RED and t[0, 0, 1] == 0


def test_a_brick_outside_the_shape_is_rejected_but_a_near_miss_is_allowed():
    s = checked(box(), tolerance=1)
    assert s.reject_reason("2x4 (0,0,0)") is None
    assert s.reject_reason("2x4 (3,0,0)") is None          # pokes 1 cell past the edge: within tolerance
    assert s.reject_reason("2x4 (8,8,0)") == "outside_shape"
    assert checked(box(), tolerance=0).reject_reason("2x4 (3,0,0)") == "outside_shape"


def test_the_model_may_not_stop_before_the_shape_is_covered():
    script = ["2x4 (0,0,0)", "", "2x4 (2,0,0)", "2x4 (0,0,1)", "2x4 (2,0,1)", ""]
    s = checked(box(x=4, y=4, z=2), min_coverage=0.9)
    res = generate(Scripted(script), "", Config(), structure=s, parse_line=parse)
    assert any(e.get("why") == "ended_early" for e in res.events)
    assert len(s.bricks) == 4 and s.metrics()["coverage"] == 1.0 and s.metrics()["iou"] == 1.0


class Candidates:
    def __init__(self, rounds):
        self.rounds, self.i = list(rounds), 0

    def candidates(self, prompt, temperature, top_k, max_tokens, k):
        r = self.rounds[self.i] if self.i < len(self.rounds) else [None] * k
        self.i += 1
        return r


def test_best_of_k_keeps_the_candidate_that_fills_the_most_of_the_shape():
    rounds = [["1x1 (0,0,0)", "2x4 (0,0,0)", "2x4 (9,9,0)"], [None, None, None]]
    s = checked(box(x=2, y=4, z=1), min_coverage=0.5)
    res = generate(Candidates(rounds), "", Config(best_of=3), structure=s, parse_line=parse)
    assert s.txt().splitlines() == ["2x4 (0,0,0)"] and res.stable


def test_photo_colors_follow_the_target_cells_each_part_covers():
    t = box(x=4, y=4, z=1)
    t[2:, :, 0] = 2  # blue half
    s = checked(t)
    for line in ("2x4 (0,0,0)", "2x4 (2,0,0)"):
        s.add(parse(line))
    assert s.colors() == [4, 1]  # LDraw red, blue


def test_a_floating_part_is_rejected_when_placed_not_rolled_back_later():
    s = checked(box(x=4, y=4, z=3))
    s.add(parse("2x4 (0,0,0)"))
    assert s.reject_reason("2x4 (2,0,1)") == "floating"      # nothing under it, nothing on it
    assert s.reject_reason("2x4 (0,0,1)") is None             # sits on the first brick


def test_repeating_a_rejected_part_raises_temperature_and_gives_up_when_stuck():
    from brickgpt.decode import Config, generate

    rounds = [["2x4 (0,0,0)"]] + [["2x4 (9,9,0)"]] * 60
    s = checked(box(x=2, y=4, z=1), min_coverage=0.0)
    res = generate(Candidates(rounds), "", Config(best_of=3, max_repeats=5, max_backtracks=0), structure=s, parse_line=parse)
    temps = [e["temperature"] for e in res.events if e["kind"] in ("reject", "repeat")]
    assert res.gave_up == "stuck repeating rejected parts" and temps == sorted(temps) and temps[-1] > temps[0] + 0.2
    assert s.txt().splitlines() == ["2x4 (0,0,0)"]


def test_a_stuck_model_backtracks_out_of_its_pattern_before_giving_up():
    from brickgpt.decode import Config, generate

    # Two good parts, then the model locks onto an outside part; after backtracking it continues differently.
    class Locked:
        def __init__(self):
            self.calls = 0

        def candidates(self, prompt, temperature, top_k, max_tokens, k):
            self.calls += 1
            placed = prompt.count("\n")
            if "backtracked" in self.__dict__:
                return ["2x4 (2,0,0)"] if placed == 0 else [None]
            if placed < 2:
                return [["2x4 (0,0,0)", "2x4 (0,0,1)"][placed]]
            self.stuck = getattr(self, "stuck", 0) + 1
            if self.stuck > 3:
                self.backtracked = True
            return ["2x4 (9,9,0)"]

    s = checked(box(x=4, y=4, z=2), min_coverage=0.0)
    res = generate(Locked(), "", Config(best_of=3, max_repeats=2), structure=s, parse_line=parse)
    assert any(e["kind"] == "backtrack" for e in res.events) and not res.gave_up


def test_stuck_again_at_the_same_point_backtracks_further():
    from brickgpt.decode import Config, generate

    # Rebuilds the same column each time and locks at the same height: the cut must grow.
    column = [f"1x1 (0,0,{z})" for z in range(4)]

    class SameLock:
        def candidates(self, prompt, temperature, top_k, max_tokens, k):
            placed = prompt.count("\n")
            return [column[placed]] if placed < 4 else ["2x4 (9,9,0)"]

    s = checked(box(x=1, y=1, z=4), min_coverage=1.1)
    res = generate(SameLock(), "", Config(best_of=3, max_repeats=1, max_backtracks=3), structure=s, parse_line=parse)
    assert [e["removed"] for e in res.events if e["kind"] == "backtrack"] == [2, 4, 4]
