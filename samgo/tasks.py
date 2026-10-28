"""Curriculum tasks: a verified build with its last k pieces removed.

The model sees the target as one text grid per layer: '.' is empty, '#' is
already built, and a color letter marks a cell it still has to fill. It
answers with one JSON piece per line. A task passes when the completed build
passes the verifier with exact occupancy and colors; any parts that do that
count, not only the removed ones.
"""

import json
import re
from dataclasses import dataclass

import numpy as np

from .builder import build
from .parts import COLOR_INDEX, FOOTPRINTS
from .targets import sample
from .verifier import MatchConfig, Report, verify

EXACT = MatchConfig(min_iou=1.0, min_color_accuracy=1.0)
LETTERS = dict(zip(COLOR_INDEX, "KBGRYWOTNLD"))  # black, blue, green, red, yellow, white, orange, tan, brown, light/dark gray
NAMES = {letter: name for name, letter in LETTERS.items()}

SYSTEM = f"""You complete LEGO-compatible builds on a stud grid.

Grid: x and y are studs, z is plate layers. A plate is 1 layer tall, a brick is 3.
Parts: plates and bricks with footprints {", ".join(f"{a}x{b}" for a, b in FOOTPRINTS)} (either orientation).
A piece at (x, y, z) with size w x l fills x..x+w-1, y..y+l-1, and z..z+h-1 (h = 1 plate, 3 brick).
Two pieces connect only when one sits directly on the other and they share a stud. Side contact does not connect.

The finished build must fill every target cell with the right color and nothing else,
have no overlaps, and be one connected structure resting on z = 0.

Answer with one JSON object per line and nothing else, for example:
{{"w": 2, "l": 4, "type": "brick", "x": 0, "y": 0, "z": 3, "color": "red"}}"""


@dataclass
class Task:
    id: str
    family: str
    k: int                 # pieces removed
    level: str             # curriculum level: "1", "2", "5", "10", "20", or "all" (full build from empty)
    target: np.ndarray
    given: list[dict]      # pieces already built
    removed: list[dict]    # the builder's own completion, for reference only

    def prompt(self) -> str:
        built = np.zeros(self.target.shape, dtype=bool)
        for p in self.given:
            h = 3 if p["type"] == "brick" else 1
            built[p["x"]:p["x"] + p["w"], p["y"]:p["y"] + p["l"], p["z"]:p["z"] + h] = True
        X, Y, Z = self.target.shape
        colors = {name: COLOR_INDEX[name] for name in LETTERS}
        legend = ", ".join(f"{LETTERS[n]} = {n}" for n in colors if (self.target == colors[n]).any())
        letter_of = {i: LETTERS[n] for n, i in colors.items()}
        layers = []
        for z in range(Z):
            rows = []
            for y in range(Y - 1, -1, -1):  # north at the top
                rows.append("".join("#" if built[x, y, z] else letter_of.get(int(self.target[x, y, z]), ".")
                                    for x in range(X)))
            layers.append(f"z={z}\n" + "\n".join(rows))
        todo = int(((self.target > 0) & ~built).sum())
        return (f"Target size: x={X}, y={Y}, z={Z}. Each layer is printed with y={Y - 1} on the first row and "
                f"y=0 on the last, x=0 on the left.\n"
                f"'.' = empty, '#' = already built, letters = cells you must fill ({legend}).\n\n"
                + "\n\n".join(layers)
                + f"\n\n{todo} cells are left to fill. Output the pieces that fill exactly those cells.")


PIECE = re.compile(r"\{[^{}]*\}")


def parse(answer: str) -> list:
    """Every JSON object in the answer, in order; malformed ones are kept as raw text so they fail."""
    out = []
    for m in PIECE.finditer(answer):
        try:
            out.append(json.loads(m.group()))
        except json.JSONDecodeError:
            out.append(m.group())
    return out


def score(task: Task, answer: str) -> Report:
    added = parse(answer)
    if not added:
        return verify([], task.target, config=EXACT)
    return verify(task.given + added, task.target, config=EXACT)


def make_tasks(seeds: range, levels: tuple) -> list[Task]:
    """One task per (seed, level) whose builder solution passes exactly.

    A numeric level k removes the last k pieces (the build must have more than k);
    "all" removes every piece, so the model builds the whole target from empty.
    """
    tasks = []
    for seed in seeds:
        family, target = sample(seed)
        solution = build(target, seed=seed)
        if not solution.report.passed or solution.report.metrics.get("iou", 1.0) < 1.0:
            continue
        n = len(solution.pieces)
        for level in levels:
            k = n if level == "all" else int(level)
            if level != "all" and n <= k:
                continue
            tasks.append(Task(f"{seed}-{level}", family, k, str(level), target, solution.pieces[:n - k],
                              solution.pieces[n - k:]))
    return tasks


if __name__ == "__main__":
    task = next(t for t in make_tasks(range(20), (2,)) if t.target.size < 400)
    print(SYSTEM, "\n\n---\n")
    print(task.prompt())
    print("\n--- reference answer ---")
    answer = "\n".join(json.dumps(p) for p in task.removed)
    print(answer)
    print("--- verifier:", score(task, answer).passed)
