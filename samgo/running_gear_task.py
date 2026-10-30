"""The running-gear task: Glimmer makes a car body roll by editing it (spec.md section 0, task 1).

One definition used three ways, so training and use always match:
- `gen_running_gear` turns verified bases (gen_bases) into SFT rows;
- the evaluation and the demo loop pose the same problem to Glimmer and check its answer the same way.

The problem: a verified car (arched builder) with the running gear of one or both axles taken out, its Technic
beams turned back into plain bricks. Code supplies where each tire is (its center along the car and its outline
diameter); Glimmer chooses the parts. It sees only the parts at chassis level (the rest of the car is unchanged).
"""

import json
import random
import re

import numpy as np

from .stability import stability
from .technic import (AXLE_LENGTHS, HOLE_HEIGHT, PLATE, STOP_AXLE_LENGTHS, WHEELS, structural, verify_function,
                      wheel_studs)
from .verifier import MatchConfig, verify

WHEEL_LIST = "; ".join(f'"{k}" ({v[0]} mm across, {wheel_studs(k)} studs wide)' for k, v in WHEELS.items())

RULES = f"""You are a LEGO Technic engineer. You edit a LEGO model given as a numbered list of parts (JSON).

Coordinates: x and y in studs (1 stud = 8 mm), z in plates (1 plate = 3.2 mm; a brick is 3 plates tall).
A part {{"w": W, "l": L, ...}} fills studs x..x+W-1 along x, y..y+L-1 along y, from height z.
Part types:
- {{"type": "plate"|"brick", "w", "l", "x", "y", "z", "color"}}: plain parts, joined only by studs (one part sitting
  on another, sharing at least one stud).
- {{"type": "technic_brick", "w": N, "l": 1, "x", "y", "z", "color", "hole": "round"}}: a 1xN brick (N = 2, 4, 6, 8)
  with N-1 round holes across it (along y), at the stud edges x+1 ... x+N-1, {HOLE_HEIGHT} mm above its bottom.
- {{"type": "axle", "axis": "y", "x": E, "z": Z, "y": Y0, "length": L}}: a rod along y on stud edge x=E, at the hole
  height of Technic bricks whose bottom is at plate z=Z, spanning y=Y0..Y0+L. Real lengths only:
  {", ".join(map(str, AXLE_LENGTHS))}. It must pass through at least 2 aligned round holes and no other part.
  A stop axle ("stop": true; lengths {", ".join(map(str, STOP_AXLE_LENGTHS))}) may carry a single wheel;
  any other axle needs a wheel on each end.
- {{"type": "wheel", "axis": "y", "x": E, "z": Z, "y": Y0, "size": S}}: a wheel with tyre on that axle, starting at
  y=Y0. Sizes: {WHEEL_LIST}. It must not hit any part when it turns, and the wheels must be the lowest thing
  touching the ground.
A car rolls when it has at least 2 parallel axle lines, each with a wheel on each side, turning in round holes."""

TASK = """This car body's length runs along x, its width along y. Below are its parts at chassis level near the
tires (the rest of the car stays as it is).

Add running gear for these tires (wheel arches are already cut into the body):
{tires}

Make the car roll: change plain bricks into Technic bricks where an axle must pass, and add axles and wheels.
Answer with a JSON list of edits, and nothing after it:
[{{"op": "set", "index": I, "part": {{...}}}}, {{"op": "add", "part": {{...}}}}, {{"op": "remove", "index": I}}]

Parts:
{parts}"""

FEEDBACK = """The verifiers checked your edited model:
{issues}
Fix these problems. Answer with the complete JSON list of edits again (it replaces your previous answer; edits
still refer to the original numbered parts)."""

GEAR = ("axle", "wheel")


def axles_of(parts: list[dict]) -> list[int]:
    """Stud edges (x) of the car's axle lines, front to back."""
    return sorted({p["x"] for p in parts if p.get("type") == "axle"})


def pose(parts: list[dict], target: np.ndarray, strip: list[int]) -> dict:
    """The problem for a verified car with the running gear on axle lines `strip` (stud edges) taken out.

    Returns the full input parts, the local view (indices into it) Glimmer sees, the tire text, and the
    reference answer as edits on the local numbering.
    """
    def on(p, e):  # a Technic beam or gear part that belongs to axle line e
        return (p.get("type") in GEAR and p["x"] == e) or (p.get("type") == "technic_brick" and p["x"] < e < p["x"] + p["w"])

    full, answer_parts, beams = [], [], []
    for p in parts:
        e = next((e for e in strip if on(p, e)), None)
        if e is None:
            full.append(p)
        elif p["type"] == "technic_brick":
            beams.append((len(full), p))
            full.append({k: v for k, v in p.items() if k != "hole"} | {"type": "brick"})
        else:
            answer_parts.append(p)
    rows = {(e, p["z"]) for e in strip for _, p in beams if p["x"] < e < p["x"] + p["w"]}  # each axle at its height

    def near(p):
        return any(p["x"] <= e + 5 and p["x"] + p["w"] >= e - 5 and z - 3 <= p["z"] <= z + 3 for e, z in rows)

    local = [i for i, p in enumerate(full) if p.get("type") not in GEAR and near(p)]
    pos = {i: k for k, i in enumerate(local)}
    answer = [{"op": "set", "index": pos[i], "part": p} for i, p in beams]
    answer += [{"op": "add", "part": p} for p in answer_parts]
    tires = []
    for e in strip:
        wheels = [p for p in answer_parts if p["type"] == "wheel" and p["x"] == e]
        size = wheels[0]["size"]
        tires.append(f"- tire centered on stud edge x={e}, outline about {WHEELS[size][0]:.0f} mm across; body spans "
                     f"y={min(wheels, key=lambda w: w['y'])['y']}..{max(w['y'] + wheel_studs(w['size']) for w in wheels) - 1} "
                     "there")
    return {"full": full, "local": local, "tires": "\n".join(tires), "answer": answer, "target": target}


def prompt(problem: dict) -> list[dict]:
    numbered = "\n".join(f"{k}: {json.dumps(problem['full'][i])}" for k, i in enumerate(problem["local"]))
    return [{"role": "system", "content": RULES},
            {"role": "user", "content": TASK.format(tires=problem["tires"], parts=numbered)}]


def apply(problem: dict, edits: list[dict]) -> list[dict]:
    """The whole car after `edits` (on the local numbering)."""
    parts: list[dict | None] = list(problem["full"])
    added = []
    for e in edits:
        if e.get("op") == "set":
            parts[problem["local"][int(e["index"])]] = e["part"]
        elif e.get("op") == "remove":
            parts[problem["local"][int(e["index"])]] = None
        elif e.get("op") == "add":
            added.append(e["part"])
    return [p for p in parts if p is not None] + added


def check(parts: list[dict], target: np.ndarray) -> tuple[bool, list[str]]:
    """Structure (the chassis counts as standing on its wheels), function, stability; error text for Glimmer."""
    try:
        solid = structural(parts)  # Technic bricks become plain bricks here, in the same order
        anchors = frozenset(k for k, p in enumerate(q for q in parts if q.get("type") not in GEAR)
                            if p.get("type") == "technic_brick")
        s = verify(solid, target, config=MatchConfig(), anchors=anchors)
        f = verify_function(parts)
    except Exception as e:  # noqa: BLE001  a malformed part
        return False, [f"invalid model: {e}"]
    issues = [f"structure: {i.message}" for i in s.issues] + [f"function: {i.message}" for i in f.issues]
    if not issues:
        st = stability(parts, vehicle=True)
        issues += [f"stability: {i.message}" for i in st.issues]
    return not issues, issues


def edits_from(text: str) -> list[dict]:
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
    if not m:
        raise ValueError("no JSON list of edits in the answer")
    return json.loads(m.group(0))


def flawed(answer: list[dict], rng: random.Random) -> list[dict]:
    """A plausible wrong first answer, for feedback examples: one typical mistake."""
    bad = json.loads(json.dumps(answer))
    kind = rng.choice(["length", "wheel_shift", "drop_wheel", "low_axle"])
    axles = [e for e in bad if e["op"] == "add" and e["part"]["type"] == "axle"]
    wheels = [e for e in bad if e["op"] == "add" and e["part"]["type"] == "wheel"]
    if kind == "length" and axles:
        a = rng.choice(axles)["part"]
        a["length"] = next((n for n in range(a["length"] + 1, 40) if n not in AXLE_LENGTHS), a["length"] + 1)
    elif kind == "wheel_shift" and wheels:
        rng.choice(wheels)["part"]["y"] += rng.choice([-1, 1])
    elif kind == "drop_wheel" and wheels:
        bad.remove(rng.choice(wheels))
    elif axles:
        for a in axles:
            a["part"]["z"] += 3
    return bad
