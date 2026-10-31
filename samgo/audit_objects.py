"""Frozen assembly/repair probes across object families, with native mechanism checks.

Diagnostic development cases, not an unseen benchmark. Reference answers are kept out of model messages.
"""

import copy
import json
from pathlib import Path

import numpy as np

from .assembly import digest, save
from .edits import apply_edits
from .parts import COLOR_INDEX, HEIGHTS, parse_piece
from .running_gear_task import TASK_FRAME, _check_issues, apply, check, explain, pose_frame, prompt
from .technic import FUNCTIONAL, _validate, structural, verify_function
from .verifier import verify


def grid(parts):
    solid = structural(parts)
    shape = tuple(
        max(p[k] + (p["w"] if k == "x" else p["l"] if k == "y" else HEIGHTS[p["type"]]) for p in solid)
        for k in ("x", "y", "z")
    )
    target = np.zeros(shape, dtype=np.int32)
    for p in solid:
        target[p["x"] : p["x"] + p["w"], p["y"] : p["y"] + p["l"], p["z"] : p["z"] + HEIGHTS[p["type"]]] = (
            COLOR_INDEX[p["color"]]
        )
    return target


def brick(w, length, x, y, z, kind="plate", color="red"):
    return dict(type=kind, w=w, l=length, x=x, y=y, z=z, color=color)


def strip(x0, x1, y, z, width=1, color="red", first=None):
    out = []
    while x0 < x1:
        n = first if first and first <= x1 - x0 else max(n for n in (1, 2, 3, 4, 6, 8) if n <= x1 - x0)
        first = None
        out.append(brick(n, width, x0, y, z, color=color))
        x0 += n
    return out


def rolling_reference(span=6, axles=2, offset=0, train=False):
    """Isolated end pads force support span; outside guards reject unnecessarily long beams."""
    edges = [offset + 6 + 12 * k for k in range(axles)]
    end = edges[-1] + 6
    parts = strip(offset, end, 2, 6, color="dark_gray") + strip(offset, end, 2, 7, width=2, first=4)
    for edge in edges:
        parts.append(brick(2, 2, edge - 1, 2, 5))
        for x in (edge - span // 2, edge + span // 2 - 1):
            parts.append(brick(1, 1, x, 3, 5, color="yellow"))
        for x in (edge - span // 2 - 1, edge + span // 2):
            parts += [brick(1, 2, x, 2, 2, "brick", "dark_gray"), brick(1, 2, x, 2, 5, color="dark_gray")]
        for y in (2, 3):
            parts.append(brick(span, 1, edge - span // 2, y, 2, "technic_brick", "black") | {"hole": "round"})
        parts += [dict(type="axle", axis="y", x=edge, y=0, z=2, length=6)]
        parts += [dict(type="wheel", axis="y", x=edge, y=y, z=2, size="30.4x14") for y in (0, 4)]
    if train:
        # Procedural locomotive silhouette; the model's task is its multi-axle running gear.
        parts += strip(offset, end, 2, 8, width=2, color="blue")
        for z in (9, 12):
            parts.append(brick(4, 2, offset + 1, 2, z, "brick", "blue"))
        parts.append(brick(6, 2, offset, 2, 15, color="black"))
        parts.append(brick(2, 2, offset + 8, 2, 9, "brick", "black"))
    return parts, edges


def bed_reference():
    parts = []
    for x, y in ((0, 0), (10, 0), (0, 6), (10, 6)):
        for z in (0, 3):
            parts.append(brick(2, 2, x, y, z, "brick", "reddish_brown"))
    for y in range(0, 8, 2):
        parts += strip(0, 12, y, 6, width=2, color="blue", first=6)
        parts += strip(0, 12, y, 7, width=2, color="blue", first=4)
    for z in (8, 11):
        parts.append(brick(2, 8, 0, 0, z, "brick", "reddish_brown"))
    return parts


def moving_groups(parts, kinds):
    from .kinematics import analyse

    bodies, _ = analyse(parts)
    groups = []
    for i, p in enumerate(parts):
        if p["type"] in kinds:
            members = next(sorted(v) for v in bodies.values() if i in v)
            if members not in groups:
                groups.append(members)
    return groups


def bound_motion(parts, members, degrees, axis):
    from .kinematics import bound_motion as check_motion

    return check_motion(parts, members, degrees, axis)


def native_checks(case, parts):
    family = case["family"]
    ctx = case.get("context", {})
    if family in ("chair", "skateboard"):
        from .chair import checks

        out = checks(parts, ctx["requirements"], set(ctx["seat"]) if "seat" in ctx else None)
        if family == "skateboard":
            from .skateboard import steer

            fn = verify_function(parts)
            out["rolls"] = (fn.passed, "; ".join(i.message for i in fn.issues))
            out["steer"] = steer(parts, ctx["steer"])
        return out
    if family == "bicycle":
        from . import bicycle as b
        from .kinematics import requirement_met

        return dict(
            structure=b._structure(parts),
            connected=b._connected(parts),
            running_gear=b._running_gear(parts),
            spin=requirement_met(parts, b.REQUIREMENTS[0]),
            wheels_in_line=b._in_line(parts),
            steer=b._steering(parts, 30),
            crank=b._crank(parts),
            stands=b.balance(parts)[:2],
        )
    if family in ("door", "microwave", "tv", "laptop"):
        from .doors import checks

        result = checks(
            parts,
            ctx["requirements"],
            {},
            vehicle=ctx.get("vehicle", False),
        )
        for key, (members, degrees) in ctx["moving"].items():
            result[key] = bound_motion(
                parts, members, degrees, "horizontal" if family == "laptop" else "vertical"
            )
        return result
    if family == "bed":
        from .stability import stability

        r = verify(parts, np.array(case["target"], dtype=np.int32))
        st = stability(parts) if r.passed else None
        feet = all(
            any(
                p["z"] == 0 and p["x"] <= x < p["x"] + p["w"] and p["y"] <= y < p["y"] + p["l"] for p in parts
            )
            for x, y in ((0, 0), (10, 0), (0, 6), (10, 6))
        )
        return {
            "structure": (r.passed, "; ".join(i.message for i in r.issues)),
            "stands": (bool(st and st.passed), "; ".join(i.message for i in st.issues) if st else "deferred"),
            "four_feet": (feet, "ground supports required at all four bed corners"),
        }
    raise ValueError(f"unsupported family {family}")


def inspect_case(case, edits):
    """Return bounded check results without interpreting a model's prose as success."""
    try:
        if case["mode"] == "gear":
            pr = problem(case)
            candidate = apply(pr, edits)
            ok, messages = explain(pr, edits, partial=case.get("partial", False))
            codes = sorted(
                {
                    i.code
                    for _, i in _check_issues(
                        candidate,
                        pr["target"],
                        partial_edge=pr.get("edge") if case.get("partial") else None,
                        changed=frozenset(range(len(pr["full"]), len(candidate))),
                    )
                }
            )
            missing = [
                e
                for e in case.get("required_edges", [])
                if len([p for p in candidate if p["type"] == "wheel" and p["axis"] == "y" and p["x"] == e])
                < 2
            ]
            if missing:
                ok = False
                codes.append("missing_required_axles")
                messages.append(f"required wheel pairs absent at x={missing}")
        else:
            allowed = frozenset({"add"}) if case["mode"] == "add" else frozenset({"set"})
            candidate = apply_edits(case["input"], edits, allowed_ops=allowed).parts
            if case["mode"] == "repair":
                for old, new in zip(case["input"], candidate):
                    if set(old) != set(new) or any(
                        new[k] != old[k] for k in ("type", "color", "role") if k in old
                    ):
                        raise ValueError("repair must preserve part identity, fields, color and role")
            for p in candidate:
                if p["type"] in FUNCTIONAL and (err := _validate(p)):
                    raise ValueError(err)
                if p["type"] in ("plate", "brick"):
                    parse_piece(p | {"color": "black"})  # specialized palettes include transparent colors
            checks = native_checks(case, candidate)
            failed = {k: v for k, v in checks.items() if not (v if isinstance(v, bool) else v[0])}
            ok = not failed
            codes = sorted(failed)
            messages = [f"{k}: {v[1] if not isinstance(v, bool) else 'failed'}" for k, v in failed.items()]
        return dict(
            passed=bool(ok),
            codes=codes,
            issues=messages,
            candidate=candidate,
            beams=[p.get("w") for p in candidate if p["type"] == "technic_brick"],
        )
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as e:
        return dict(passed=False, codes=["invalid_edit_or_part"], issues=[str(e)], candidate=None, beams=[])


def problem(case):
    return dict(
        full=case["input"],
        local=case["local"],
        target=np.array(case["target"], dtype=np.int32),
        task=TASK_FRAME,
        tires=case["tires"],
        **({"edge": case["edge"]} if "edge" in case else {}),
    )


def messages(case):
    if case["mode"] == "gear":
        chat = prompt(problem(case))
        if case["family"] == "train":
            chat[-1]["content"] += (
                "\nThis is a rigid train-style rolling locomotive. Complete every listed axle line, "
                + f"including all {len(case['required_edges'])} wheel pairs. Rail compatibility and propulsion are outside this task."
            )
        return chat
    mode = case["mode"]
    rules = (
        "Coordinates: x/y are studs, z is plates; brick height 3, plate height 1. Parts connect by vertically shared "
        + "studs or matching connectors. Axles/wheels share their axis line and height; axles need aligned holes. "
        + "Hinge halves must have matching pin coordinates/axis; turntable halves share x/y with top one plate above base. "
        + "Only real catalog dimensions are allowed. Return only a JSON list of edits. "
    )
    if mode == "repair":
        rules += 'Use only {"op":"set","index":I,"part":{...}} on original numbering. Preserve every part type, field, color and role. '
        task = "Repair this assembly while preserving its object and required motions. Prefer the smallest necessary change."
    else:
        rules += 'Use only {"op":"add","part":{...}}. Bricks/plates have w,l,x,y,z,color; footprints are 1x1/2/3/4/6/8 or 2x2/3/4/6/8. '
        task = "Complete this bed by adding supports under the mattress. It must stand on four corners (0,0),(10,0),(0,6),(10,6); preserve its existing parts."
    return [
        {"role": "system", "content": rules},
        {
            "role": "user",
            "content": task
            + "\nObject: "
            + case["family"]
            + "\nRequirements: "
            + json.dumps(case.get("context", {}).get("requirements", []))
            + "\nParts:\n"
            + "\n".join(f"{i}: {json.dumps(p)}" for i, p in enumerate(case["input"])),
        },
    ]


def gear_case(name, parts, edges, family="car"):
    target = grid(parts)
    pr = pose_frame(parts, target, edges)
    if pr is None:
        raise ValueError(f"{name} reference invalid: {check(parts, target)}")
    return dict(
        id=name,
        family=family,
        mode="gear",
        input=pr["full"],
        reference=parts,
        target=target.tolist(),
        local=pr["local"],
        tires=pr["tires"],
        oracle=pr["answer"],
        required_edges=edges,
    )


def repair_case(name, family, parts, index, changes, context):
    broken = copy.deepcopy(parts)
    indices = [index] if isinstance(index, int) else index
    for i in indices:
        broken[i].update(changes)
    return dict(
        id=name,
        family=family,
        mode="repair",
        input=broken,
        reference=parts,
        context=context,
        oracle=[{"op": "set", "index": i, "part": parts[i]} for i in indices],
    )


def make_cases(out: Path, *, include_local_fixtures=False, single_axle=False):
    from .bicycle import build_bicycle
    from .chair import build_chair
    from .doors import build_microwave, CAR_REQUIREMENTS
    from .skateboard import build_skateboard

    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for span in (4, 6, 8):
        parts, edges = rolling_reference(span)
        cases.append(gear_case(f"car_span_{span}", parts, edges))
    parts, edges = rolling_reference(6, offset=5)
    cases.append(gear_case("car_span_6_translated", parts, edges))
    for n in (4, 6):
        parts, edges = rolling_reference(6, axles=n, train=True)
        cases.append(gear_case(f"train_{n}_axles", parts, edges, "train"))
    parts = bed_reference()
    cases.append(
        dict(
            id="bed_four_legs",
            family="bed",
            mode="add",
            input=[p for p in parts if p["z"] >= 6],
            reference=parts,
            target=grid(parts).tolist(),
            oracle=[{"op": "add", "part": p} for p in parts if p["z"] < 6],
        )
    )
    for family, builder in (
        ("chair", build_chair),
        ("bicycle", build_bicycle),
        ("skateboard", build_skateboard),
    ):
        r = builder()
        parts = r["parts"]
        ctx = {"requirements": r["requirements"], "unsupported": r.get("unsupported", [])}
        if family == "chair":
            i = max(i for i, p in enumerate(parts) if p["type"] == "turntable_top")
            ctx["seat"] = list(range(i, len(parts)))
        if family == "skateboard":
            ctx["requirements"] = [q for q in r["requirements"] if q["kind"] == "spin"]
            ctx["steer"] = r["requirements"][1]
        i = next(i for i, p in enumerate(parts) if p["type"] == "wheel")
        cases.append(
            repair_case(f"{family}_wheel_alignment", family, parts, i, {"z": parts[i]["z"] + 1}, ctx)
        )
        if family == "chair":
            i = max(i for i, p in enumerate(parts) if p["type"] == "turntable_top")
            cases.append(repair_case("chair_seat_pivot", family, parts, i, {"x": parts[i]["x"] + 1}, ctx))
        if family == "bicycle":
            i = next(i for i, p in enumerate(parts) if p.get("axle_hole"))
            cases.append(
                repair_case(
                    "bicycle_crank_connection",
                    family,
                    parts,
                    i,
                    {"axle_hole": "end" if parts[i]["axle_hole"] == "start" else "start"},
                    ctx,
                )
            )
    r = build_microwave()
    parts = r["parts"]
    groups = moving_groups(parts, {"hinge_top"})
    dish = moving_groups(parts, {"turntable_top"})[0]
    ctx = dict(
        requirements=r["requirements"],
        moving={f"door_{i}": (g, 90) for i, g in enumerate(groups)} | {"dish": (dish, 360)},
    )
    for kind, label in (("hinge_top", "door_hinge"), ("turntable_top", "dish_pivot")):
        i = next(i for i, p in enumerate(parts) if p["type"] == kind)
        changes = {"pin_x": parts[i]["pin_x"] + 1} if kind == "hinge_top" else {"x": parts[i]["x"] + 1}
        indices = [k for k, p in enumerate(parts) if p["type"] == "hinge_top"] if kind == "hinge_top" else i
        cases.append(repair_case("microwave_" + label, "microwave", parts, indices, changes, ctx))
    for name in ("camry", "beetle") if include_local_fixtures else ():
        path = Path(f"runs/demo/doors/{name}_doors.json")
        if not path.exists():
            raise FileNotFoundError(f"requested local door fixture unavailable: {path}")
        if path.exists():
            parts = json.loads(path.read_text())
            parts = parts["parts"] if isinstance(parts, dict) else parts
            groups = moving_groups(parts, {"hinge_top"})
            ctx = dict(
                requirements=CAR_REQUIREMENTS,
                vehicle=True,
                moving={f"door_{i}": (g, 90) for i, g in enumerate(groups)},
            )
            i = next(i for i, p in enumerate(parts) if p["type"] == "hinge_top")
            cases.append(
                repair_case(
                    name + "_door_hinge",
                    "door",
                    parts,
                    [k for k in groups[0] if parts[k]["type"] == "hinge_top"],
                    {"pin_x": parts[i]["pin_x"] + 1},
                    ctx,
                )
            )
    # Two compact, valid-part pivot examples, independent of photo fixtures.
    tv = [
        brick(8, 2, 0, 0, 0, color="black"),
        dict(type="turntable_base", w=2, l=2, x=3, y=0, z=1, color="black"),
        dict(type="turntable_top", w=2, l=2, x=3, y=0, z=2, color="black"),
        brick(8, 1, 0, 0, 3, "brick", "blue"),
    ]
    laptop = [
        brick(4, 2, 0, 0, 0),
        brick(4, 2, 0, 2, 0),
        brick(4, 2, 0, 1, 1),
        dict(type="click_a", w=4, l=1, x=0, y=3, z=1, pin_x=0, pin_y=4, pin_axis="x", color="black"),
        dict(type="click_b", w=4, l=1, x=0, y=4, z=1, pin_x=0, pin_y=4, pin_axis="x", color="black"),
        brick(4, 1, 0, 4, 2, "brick", "blue"),
    ]
    for name, parts, kind, axis in (
        ("tv", tv, "turntable_top", "vertical"),
        ("laptop", laptop, "click_b", "horizontal"),
    ):
        ctx = dict(
            requirements=[dict(name="screen_moves", kind="swing", axis=axis, degrees=60)],
            moving={"screen": (moving_groups(parts, {kind})[0], 60)},
        )
        i = next(i for i, p in enumerate(parts) if p["type"] == kind)
        changes = {"x": parts[i]["x"] + 1} if name == "tv" else {"pin_y": parts[i]["pin_y"] + 1}
        cases.append(repair_case(name + "_pivot", name, parts, i, changes, ctx))
    # Freeze the exact previous failures, avoiding new perception or body generation.
    for name in ("pipeline-live-camry", "pipeline-fresh-camry") if include_local_fixtures else ():
        s = json.loads((Path("runs/audit") / name / "summary.json").read_text())
        trace_path = Path(s["artifacts"]["trace"])
        trace = json.loads(trace_path.read_text())
        step = trace["steps"][-1]
        target = np.load(trace_path.parent.parent / "assembly_target.npz")["target"]
        ref = json.loads((trace_path.parent / "reference.json").read_text())
        edge = step["edge"]
        selected = [
            p
            for p in ref
            if p["type"] in ("axle", "wheel", "bush")
            and p["x"] == edge
            or p["type"] == "technic_brick"
            and p["x"] < edge < p["x"] + p["w"]
        ]
        cases.append(
            dict(
                id=name,
                family="car",
                mode="gear",
                input=step["input"],
                reference=ref,
                target=target.tolist(),
                local=step["local"],
                tires=step["tires"],
                edge=edge,
                partial=step["partial"],
                oracle=[{"op": "add", "part": p} for p in selected],
            )
        )
    if single_axle:
        cases = [
            gear_case(c["id"] + "_single", c["reference"], c["required_edges"][:1])
            | {"required_edges": c["required_edges"]}
            for c in cases
            if c["id"].startswith("car_span_")
        ]
    admission = []
    for case in cases:
        oracle = inspect_case(case, case["oracle"])
        bad = inspect_case(case, [])
        admission.append(
            dict(
                id=case["id"],
                oracle_passed=oracle["passed"],
                oracle_issues=oracle["issues"],
                input_passed=bad["passed"],
                input_issues=bad["issues"],
            )
        )
    save(out / "admission.json", admission)
    failed = [a for a in admission if not a["oracle_passed"] or a["input_passed"]]
    if failed:
        raise ValueError(f"case admission failed: {failed}")
    save(out / "cases.json", {"sha256": digest(cases), "cases": cases})
    return cases
