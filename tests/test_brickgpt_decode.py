from brickgpt.decode import Config, generate
from brickgpt.structure import Structure, from_txt, parse


class Scripted:
    """Fake model: answers lines from a script, one per call; '' ends the answer."""

    def __init__(self, lines):
        self.lines, self.i = list(lines), 0

    def line(self, prompt, temperature, top_k, max_tokens):
        text = self.lines[self.i] if self.i < len(self.lines) else ""
        self.i += 1
        return text, "stop"


BASE = ["2x4 (0,0,0)", "2x4 (2,0,0)", "2x4 (0,0,1)"]


def test_bad_lines_are_rejected_and_good_ones_kept():
    script = ["2x4 (0,0,0)", "hello", "3x3 (5,5,0)", "2x4 (19,0,0)", "2x4 (0,0,0)", "2x4 (2,0,0)", "2x4 (0,0,1)", ""]
    res = generate(Scripted(script), "", Config())
    assert res.structure.txt().splitlines() == BASE
    assert [e["why"] for e in res.events if e["kind"] == "reject"] == ["malformed", "not_in_library", "out_of_bounds",
                                                                         "collision"]
    assert res.stable and res.rollbacks == 0


def test_unstable_tail_rolls_back_and_regenerates():
    # A 1x8 brick hanging off one stud of a single 1x1 column overloads that stud; the second pass ends cleanly.
    tower = [f"1x1 (0,0,{z})" for z in range(3)]
    script = tower + ["8x1 (0,0,3)", "", ""]
    res = generate(Scripted(script), "", Config())
    assert res.rollbacks == 1 and res.stable
    assert res.structure.txt().splitlines() == tower


def test_dataset_structure_round_trips_through_ldraw_lines():
    s = from_txt("2x4 (0,0,0)\n4x2 (3,3,1)")
    ldr = s.ldr("t").splitlines()
    # Same placement as BrickGPT's Brick.to_ldr: center in LDU, -y up, turned matrix for 4x2.
    assert ldr[3] == "1 71 20 0 40 0 0 1 0 1 0 -1 0 0 3001.dat"
    assert ldr[5] == "1 71 100 -24 80 -1 0 0 0 1 0 0 0 -1 3001.dat"


def test_v2_rules_plug_into_the_same_decoder():
    from brickgpt.v2 import Structure2
    from samgo.palette import parse as parse2

    script = ["brick 2x4 (0,0,0)", "slope 2x2 (0,0,3)", "slope 2x2 (0,0,3) -y", "tile 2x2 (0,2,3)",
              "plate 2x1 (0,0,6)", ""]
    res = generate(Scripted(script), "", Config(), structure=Structure2(), parse_line=parse2)
    # The facing-less slope is rejected; a plate only over the slope's low row (no studs there) floats and
    # is rolled back. (A 2x2 plate would hold: it reaches the high row's studs.)
    assert [e["why"] for e in res.events if e["kind"] == "reject"] == ["not_in_library"]
    assert res.rollbacks == 1 and res.stable
    assert res.structure.txt().splitlines() == ["brick 2x4 (0,0,0)", "slope 2x2 (0,0,3) -y", "tile 2x2 (0,2,3)"]
