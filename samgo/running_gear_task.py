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

TASK_FRAME = """This car is built as a frame and a shell: a plate floor, a hollow body on top, and an empty layer under
the floor (only a rim) where the running gear goes. Its length runs along x, its width along y. Below are its parts
near the tires (the rest of the car stays as it is).

Add running gear for these tires (wheel arches are already cut into the body):
{tires}

Make the car roll: hang Technic bricks under the floor where an axle must pass (they hold by sharing studs with the
floor above), and add axles, wheels and bushes. Nothing needs to be removed.
Answer with a JSON list of edits, and nothing after it:
[{{"op": "add", "part": {{...}}}}, ...]

Parts:
{parts}"""

FEEDBACK = """The verifiers checked your edited model:
{issues}
Fix these problems. Answer with the complete JSON list of edits again (it replaces your previous answer; edits
still refer to the original numbered parts)."""

GEAR = ("axle", "wheel", "bush")


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
            {"role": "user", "content": problem.get("task", TASK).format(tires=problem["tires"], parts=numbered)}]


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


def pose_open(parts: list[dict], target: np.ndarray, strip: list[int], _mirror: bool = True,
              _margin: int = 2) -> dict | None:
    """The harder problem: the body is tiled by the builder with no running gear at all, so nothing in it marks
    where the Technic beams go; the prompt gives only each tire's center along the car and its outline size.

    The reference answer removes the parts in the beams' way, adds the beams, refills the cells those parts covered
    outside the beams, and adds the axles, wheels and bushes. None if the refill cannot be laid out.
    """
    from .builder import _layout

    def on(p, e):
        return (p.get("type") in GEAR and p["x"] == e) or \
            (p.get("type") == "technic_brick" and p["x"] < e < p["x"] + p["w"])

    beams = [p for p in parts if p.get("type") == "technic_brick" and any(on(p, e) for e in strip)]
    gear = [p for p in parts if p.get("type") in GEAR and any(on(p, e) for e in strip)]
    keep = [p for p in parts if p not in beams and p not in gear]
    from .parts import COLOR_INDEX

    # Re-tile only the chassis band around the stripped axle lines (the beam layer, 2 studs either side of the
    # beams, the whole width), keeping the rest of the verified body as built: the band's new parts sit between
    # unchanged layers, so the body still holds together, and nothing in it marks where the beams go.
    hgt = {"plate": 1, "brick": 3, "technic_brick": 3}
    band = np.zeros(target.shape, bool)
    for b in beams:
        band[max(b["x"] - _margin, 0):b["x"] + b["w"] + _margin, :, b["z"]:b["z"] + 3] = True

    def cells(p):
        return (slice(p["x"], p["x"] + p["w"]), slice(p["y"], p["y"] + p["l"]), slice(p["z"], p["z"] + hgt[p["type"]]))

    solid = [p for p in keep if p.get("type") in hgt]
    in_band = [p for p in solid if band[cells(p)].any()]
    outside = [p for p in solid if p not in in_band]
    region = np.zeros_like(target)
    for p in in_band + beams:
        region[cells(p)] = COLOR_INDEX.get(p.get("color"), COLOR_INDEX["black"])
    fixed = [p | {"type": "brick"} if p.get("type") == "technic_brick" else p for p in in_band
             if p.get("type") == "technic_brick"]  # the other axle line's beams stay
    mirror = _mirror and bool((region == region[:, ::-1, :]).all())
    tiled = _layout(region, None, bricks=True, given=fixed, mirror=mirror)
    tiled = [p | {"type": "technic_brick", "hole": "round"} if k < len(fixed) else p for k, p in enumerate(tiled)]
    body_target = region
    full = outside + tiled + [p for p in keep if p.get("type") in GEAR]
    # Which input parts are in the beams' way, and what refills the rest of their cells.
    beam_cells = np.zeros(target.shape, bool)
    for b in beams:
        beam_cells[b["x"]:b["x"] + b["w"], b["y"]:b["y"] + b["l"], b["z"]:b["z"] + 3] = True
    h = {"plate": 1, "brick": 3, "technic_brick": 3}
    hit = [i for i, p in enumerate(full) if p.get("type") in h and
           beam_cells[p["x"]:p["x"] + p["w"], p["y"]:p["y"] + p["l"], p["z"]:p["z"] + h[p["type"]]].any()]
    if any(full[i].get("type") == "technic_brick" for i in hit):
        return None
    refill = np.zeros_like(target)
    for i in hit:
        p = full[i]
        sl = (slice(p["x"], p["x"] + p["w"]), slice(p["y"], p["y"] + p["l"]), slice(p["z"], p["z"] + h[p["type"]]))
        refill[sl] = np.where(beam_cells[sl], 0, body_target[sl])
    fillers = _layout(refill, None, bricks=True) if refill.any() else []
    zs = {(e, b["z"]) for e in strip for b in beams if b["x"] < e < b["x"] + b["w"]}

    def near(p):
        return any(p["x"] <= e + 5 and p["x"] + p["w"] >= e - 5 and z - 3 <= p["z"] <= z + 3 for e, z in zs)

    local = [i for i, p in enumerate(full) if p.get("type") not in GEAR and near(p)]
    if not set(hit) <= set(local):
        return None
    pos = {i: k for k, i in enumerate(local)}
    answer = [{"op": "remove", "index": pos[i]} for i in sorted(hit)]
    answer += [{"op": "add", "part": b} for b in beams] + [{"op": "add", "part": f} for f in fillers]
    answer += [{"op": "add", "part": g} for g in gear]
    tires = [f"- tire centered on stud edge x={e}, outline about "
             f"{WHEELS[next(g['size'] for g in gear if g['type'] == 'wheel' and g['x'] == e)][0]:.0f} mm across"
             for e in strip]
    problem = {"full": full, "local": local, "tires": "\n".join(tires), "answer": answer, "target": target}
    # The re-tiling can leave a piece held by nothing (a thin nose or tail skin of interleaved plates): then no
    # answer could pass. Only pose problems the reference answer solves; else tile without the mirror, then a
    # narrower band, which leaves thin skins as built.
    if not explain(problem, answer)[0]:
        if mirror:
            return pose_open(parts, target, strip, _mirror=False, _margin=_margin)
        return pose_open(parts, target, strip, _margin=_margin - 1) if _margin > 0 else None
    return problem


HGT = {"plate": 1, "brick": 3, "technic_brick": 3}


def describe(p: dict) -> str:
    t = p.get("type")
    if t == "axle":
        return f'{"stop " if p.get("stop") else ""}axle {p["length"]} long at x={p["x"]}, z={p["z"]}, y={p["y"]}..{p["y"] + p["length"]}'
    if t == "wheel":
        return f'{p.get("size", "30.4x14")} wheel at x={p["x"]}, z={p["z"]}, y={p["y"]}'
    if t == "bush":
        return f'bush at x={p["x"]}, z={p["z"]}, y={p["y"]}'
    return f'{t} {p.get("w")}x{p.get("l")} at (x={p.get("x")}, y={p.get("y")}, z={p.get("z")})'


def _stud_linked(p: dict, q: dict) -> bool:
    """Whether two parts hold each other by studs: footprints overlap and one sits directly on the other."""
    if p.get("type") not in HGT or q.get("type") not in HGT:
        return False
    overlap = p["x"] < q["x"] + q["w"] and q["x"] < p["x"] + p["w"] and p["y"] < q["y"] + q["l"] and q["y"] < p["y"] + p["l"]
    return overlap and (p["z"] + HGT[p["type"]] == q["z"] or q["z"] + HGT[q["type"]] == p["z"])


def explain(problem: dict, edits: list[dict], partial: bool = False) -> tuple[bool, list[str]]:
    """Check an answer and say exactly what is wrong in terms the model can act on: which of its parts overlap which
    input part (by number), which part rests on nothing, which axle or wheel fails and how."""
    removed = {int(e["index"]) for e in edits if e.get("op") == "remove"}
    set_at = {int(e["index"]): e["part"] for e in edits if e.get("op") == "set"}
    tagged = []  # (part, label)
    local_of = {i: k for k, i in enumerate(problem["local"])}
    for i, p in enumerate(problem["full"]):
        k = local_of.get(i)
        if k is not None and k in removed:
            continue
        if k is not None and k in set_at:
            tagged.append((set_at[k], f"your changed part {k} ({describe(set_at[k])})"))
        else:
            tagged.append((p, f"input part {k} ({describe(p)})" if k is not None else f"the body's {describe(p)}"))
    tagged += [(e["part"], f"your added {describe(e['part'])}") for e in edits if e.get("op") == "add"]
    parts = [p for p, _ in tagged]
    ok, issues = check(parts, problem["target"])
    if partial:  # a step that adds one axle line: the car cannot roll on one line yet, which is fine here
        issues = [i for i in issues if "turning axle line" not in i and not i.startswith("stability")]
        ok = not issues
    if ok:
        return True, []
    msgs = []
    # Overlaps, by the pair of parts involved.
    owners: dict = {}
    for n, (p, _) in enumerate(tagged):
        if p.get("type") in HGT:
            for x in range(p["x"], p["x"] + p["w"]):
                for y in range(p["y"], p["y"] + p["l"]):
                    for z in range(p["z"], p["z"] + HGT[p["type"]]):
                        owners.setdefault((x, y, z), []).append(n)
    from itertools import combinations

    pairs = sorted({pair for o in owners.values() if len(o) > 1 for pair in combinations(sorted(o), 2)})
    for a, b in pairs[:6]:
        la, lb = tagged[a][1], tagged[b][1]
        hint = ""
        for label in (la, lb):
            if label.startswith("input part "):
                hint = (": move your part - nothing in the input needs removing; Technic bricks hang right under the "
                        "floor" if problem.get("task") == TASK_FRAME
                        else f": remove {label.split(' (')[0].replace('input ', '')} first")
        msgs.append(f"overlap: {la} and {lb} fill the same cells{hint}")
    # Everything else from the verifiers, with part numbers turned into descriptions.
    for issue in issues:
        if "occupied twice" in issue:
            continue
        if "not supported from the ground" in issue:
            solid_idx = [n for n, (p, _) in enumerate(tagged) if p.get("type") not in GEAR]
            try:
                s = verify(structural(parts), problem["target"], config=MatchConfig(),
                           anchors=frozenset(k for k, n in enumerate(solid_idx) if parts[n].get("type") == "technic_brick"))
                gone = [(k, problem["full"][i]) for i, k in local_of.items() if k in removed]
                for i in next(x for x in s.issues if x.code == "floating").pieces[:4]:
                    p = parts[solid_idx[i]]
                    held = [k for k, q in gone if _stud_linked(p, q)]
                    why = (f" - it was held by input part {held[0]}, which you removed; put a part over or under it "
                           "that also reaches a part that stays (e.g. a plate across it and your new beam), or leave "
                           "it out" if held else " - put a part under or over it that shares a stud, or leave it out")
                    msgs.append(f"loose: {tagged[solid_idx[i]][1]} rests on nothing{why}")
            except Exception:  # noqa: BLE001
                msgs.append(issue)
            continue
        if "passes through solid parts" in issue:
            for ax in (p for p in parts if p.get("type") == "axle"):
                e, z0 = ax["x"], ax["z"]
                for q, label in tagged:
                    if q.get("type") not in HGT or not (q["x"] < e < q["x"] + q["w"]):
                        continue
                    if not (q["y"] < ax["y"] + ax["length"] and ax["y"] < q["y"] + q["l"]):
                        continue
                    if q["z"] <= z0 < q["z"] + HGT[q["type"]] and not (q["type"] == "technic_brick" and q["z"] == z0):
                        msgs.append(f"blocked: the axle at x={e}, z={z0} runs into {label}: only Technic bricks with "
                                    f"their bottom at z={z0} let it through")
            continue
        named = re.sub(r"(axle|wheel|part) (\d+)", lambda m: f"{m.group(1)} [{describe(parts[int(m.group(2))])}]"
                       if int(m.group(2)) < len(parts) else m.group(0), issue)
        msgs.append(named)
    return False, msgs[:12]


def realistic_mistake(answer: list[dict], rng: random.Random) -> list[dict]:
    """A wrong answer of the kinds the trained model actually makes (run 1's failures): a brick in the beams' way
    not removed, a refill piece moved or missing, a beam one stud or plate off, the second axle left half-done."""
    bad = json.loads(json.dumps(answer))
    removes = [e for e in bad if e["op"] == "remove"]
    fillers = [e for e in bad if e["op"] == "add" and e["part"]["type"] in ("brick", "plate")]
    beams = [e for e in bad if e["op"] == "add" and e["part"]["type"] == "technic_brick"]
    gear_x = sorted({e["part"]["x"] for e in bad if e["op"] == "add" and e["part"]["type"] == "axle"})
    kinds = [k for k, ok in (("keep_in_way", removes), ("move_filler", fillers), ("drop_filler", fillers),
                             ("shift_beam", beams), ("half_second_axle", len(gear_x) > 1)) if ok]
    if not kinds:
        return bad
    kind = rng.choice(kinds)
    if kind == "keep_in_way":
        for e in rng.sample(removes, min(len(removes), rng.choice([1, 1, 2, 3]))):
            bad.remove(e)
    elif kind == "move_filler":
        e = rng.choice(fillers)["part"]
        e[rng.choice(["x", "y", "z"])] += rng.choice([-1, 1])
    elif kind == "drop_filler":
        bad.remove(rng.choice(fillers))
    elif kind == "shift_beam":
        e = rng.choice(beams)["part"]
        e[rng.choice(["x", "z"])] += rng.choice([-1, 1])
    else:
        x = gear_x[-1]
        bad = [e for e in bad if not (e["op"] == "add" and e["part"].get("type") in GEAR and e["part"]["x"] == x)]
    return bad


def step_problem(full: list[dict], target: np.ndarray, edge: int, size: str, frame: bool = False) -> dict:
    """One step of building running gear: the current build (with whatever is already done), the parts near the
    axle line at stud edge `edge`, and that tire's position and size. No reference answer: the checker decides."""
    from .arches import _axle_row

    zb = _axle_row(WHEELS[size][0] / 2)
    local = [i for i, p in enumerate(full) if p.get("type") not in GEAR and p["x"] <= edge + 5 and
             p["x"] + p["w"] >= edge - 5 and zb - 3 <= p["z"] <= zb + (4 if frame else 3)]
    tires = f"- tire centered on stud edge x={edge}, outline about {WHEELS[size][0]:.0f} mm across"
    return {"full": full, "local": local, "tires": tires, "answer": [], "target": target} | (
        {"task": TASK_FRAME} if frame else {})


def pose_frame(parts: list[dict], target: np.ndarray, strip: list[int]) -> dict | None:
    """The running-gear problem on a frame-and-shell car (arches.frame_and_shell): the Technic bricks and gear of
    the axle lines at stud edges `strip` are taken out; the layer under the floor is empty apart from its rim, so
    nothing marks where they went. The reference answer only adds parts. None if it does not pass the checks."""
    def on(p, e):
        return (p.get("type") in GEAR and p["x"] == e) or \
            (p.get("type") == "technic_brick" and p["x"] < e < p["x"] + p["w"])

    out = [p for p in parts if any(on(p, e) for e in strip)]
    full = [p for p in parts if not any(on(p, e) for e in strip)]
    beams = [p for p in out if p.get("type") == "technic_brick"]
    gear = [p for p in out if p.get("type") in GEAR]
    if not beams or not gear:
        return None
    zs = {(e, b["z"]) for e in strip for b in beams if b["x"] < e < b["x"] + b["w"]}

    def near(p):
        return any(p["x"] <= e + 5 and p["x"] + p["w"] >= e - 5 and z - 3 <= p["z"] <= z + 4 for e, z in zs)

    local = [i for i, p in enumerate(full) if p.get("type") not in GEAR and near(p)]
    answer = [{"op": "add", "part": p} for p in beams + gear]
    tires = [f"- tire centered on stud edge x={e}, outline about "
             f"{WHEELS[next(g['size'] for g in gear if g['type'] == 'wheel' and g['x'] == e)][0]:.0f} mm across"
             for e in strip]
    problem = {"full": full, "local": local, "tires": "\n".join(tires), "answer": answer, "target": target,
               "task": TASK_FRAME}
    return problem if explain(problem, answer)[0] else None


def frame_mistake(answer: list[dict], rng: random.Random) -> list[dict]:
    """A wrong answer to the frame task (add-only): one of the mistakes of placing the gear itself - a Technic row
    one stud or plate off, a row or bush left out, a wrong axle length, a wheel moved or missing, the axle too high."""
    bad = json.loads(json.dumps(answer))
    beams = [e for e in bad if e["part"]["type"] == "technic_brick"]
    bushes = [e for e in bad if e["part"]["type"] == "bush"]
    kinds = ["flawed"] + ["shift_beam", "drop_beam"] * bool(beams) + ["drop_bush"] * bool(bushes)
    kind = rng.choice(kinds)
    if kind == "shift_beam":
        rng.choice(beams)["part"][rng.choice(["x", "z"])] += rng.choice([-1, 1])
    elif kind == "drop_beam":
        bad.remove(rng.choice(beams))
    elif kind == "drop_bush":
        bad.remove(rng.choice(bushes))
    else:
        bad = flawed(bad, rng)
    return bad


def frame_problem(parts: list[dict], target: np.ndarray, label: str) -> tuple[dict | None, bool]:
    """One step of the frame task, as the step-by-step loop poses it: "front" (the rear already done), "rear" (the
    front done) or "front_first" (the rear still missing, so the car cannot roll yet: checked as partial)."""
    edges = axles_of(parts)
    if len(edges) < 2:
        return None, False
    if label in ("front", "rear"):
        return pose_frame(parts, target, [edges[0] if label == "front" else edges[-1]]), False
    both = pose_frame(parts, target, edges)
    if both is None:
        return None, True
    first = [e for e in both["answer"] if e["part"]["x"] < edges[0] < e["part"]["x"] + e["part"].get("w", 0)
             or (e["part"]["type"] in GEAR and e["part"]["x"] == edges[0])]
    pr = both | {"answer": first, "tires": both["tires"].split("\n")[0]}
    return (pr if explain(pr, first, partial=True)[0] else None), True
