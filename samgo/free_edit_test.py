"""A generalization test with nothing object-specific in the code: Glimmer gets a body built from a photo and the
research step's requirements, and must make every edit itself (free parts from the body, add hinges, add wheels).

The checks are generic: each requirement as stated (spin / swing, axis, angle, count), counted over the build's
joints; every counted swinging surface must contain parts of the original body (a surface freed from the model, not
a panel invented beside it); plus structure, one connected assembly, and standing (vehicles: on their wheels).

    python -m samgo.free_edit_test runs/s6/boeing747/body40.json --object "Boeing 747" \\
        --requirements runs/s6/boeing747/test_requirements.json --model <sampler path> --label r5
"""

import argparse
import json
from pathlib import Path

from .audit_objects import PLACE_FORMATS
from .edits import apply_edits
from .running_gear_task import FEEDBACK, edits_from

FORMATS = PLACE_FORMATS + (
    '{"type":"bar_plate"|"clip_plate","w":2,"l":1,"x","y","z","facing":"+x"|"-x"|"+y"|"-y","color"} (plate 1x2 '
    "with a bar / a clip on the end it faces [60478 / 63868]: a bar plate and a clip plate at one height, facing "
    "each other with one empty stud between their ends, make a hinge about a horizontal axis across their width, "
    "half a stud into the gap); "
    '{"type":"technic_brick","w":N,"l":1,"x","y","z","color","hole":"round"} (1xN brick with round holes across '
    "it, at the stud edges, 5.6 mm above its bottom: axles go through two aligned holes). ")

RULES = ("Coordinates: x/y are studs, z is plates; a brick is 3 plates tall. Parts connect by vertically shared "
         "studs or matching connectors. Edits refer to the numbered input parts: "
         '{"op":"remove","index":I}, {"op":"set","index":I,"part":{...}}, {"op":"add","part":{...}}. '
         "Only real catalog dimensions. Answer with only the JSON list of edits. " + FORMATS)


def describe(req: dict) -> str:
    if req["kind"] == "spin":
        return f"- {req['name']}: at least {req['count']} wheels that spin on axles; the model stands on its wheels"
    return (f"- {req['name']}: {req['count']} surface(s) of the model that each swing {req['degrees']} degrees about "
            f"a {req['axis']} hinge without hitting anything")


def messages(obj: str, parts: list[dict], reqs: list[dict]) -> list[dict]:
    task = (f"Object: {obj}. Make its moving parts work, in real LEGO, by editing the model: free the moving "
            "surfaces from the body (remove or change parts), connect them with hinges, and add the wheels. Keep the "
            "object's shape. Requirements:\n" + "\n".join(describe(r) for r in reqs))
    return [{"role": "system", "content": RULES},
            {"role": "user", "content": task + "\nParts:\n" + "\n".join(f"{i}: {json.dumps(p)}"
                                                                       for i, p in enumerate(parts))}]


def check(original: list[dict], parts: list[dict], sources: list[int | None], reqs: list[dict]) -> dict:
    """Generic verdicts. `sources[i]`: the original index of candidate part i (None if added)."""
    from .kinematics import _sides, analyse, connected, locked, requirement_met, sweep
    from .mechanism_bases import PIVOT_AS, structural
    from .stability import stability
    from .technic import structural as technic_structural, verify_function
    from .verifier import verify

    out = {}
    r = verify(technic_structural(structural(parts)), None)
    rule = [i for i in r.issues if i.code not in ("floating", "disconnected")]
    out["structure"] = (not rule, "; ".join(i.message for i in rule[:4]) or "valid parts, no overlaps")
    out["connected"] = (connected(parts), "one structure through studs and joints")
    st = stability(locked(parts, PIVOT_AS), vehicle=any(p["type"] == "wheel" for p in parts))
    out["stands"] = (st.passed, "; ".join(i.message for i in st.issues) or "stands")
    bodies, joints = analyse(parts)
    swings = {"horizontal": 0, "vertical": 0}
    detail = {"horizontal": [], "vertical": []}
    for k, (a, b, kind, point, axis, partners) in enumerate(joints):
        sides = _sides(bodies, joints, k)
        if sides is None:
            continue
        small = min(sides, key=lambda g: sum(len(bodies[x]) for x in g))
        moving = [i for x in small for i in bodies[x]]
        if any(parts[i]["type"] in ("wheel", "axle", "bush") for i in moving):
            continue  # wheels are counted by the spin requirement
        if not any(sources[i] is not None for i in moving):
            continue  # an invented panel, not a surface freed from the model
        which = "vertical" if abs(axis[2]) > 0.9 else "horizontal"
        wanted = [q["degrees"] for q in reqs if q["kind"] == "swing" and q["axis"] == which]
        degrees = min(wanted) if wanted else 20
        if any(sweep(parts, moving, point, axis, s * degrees, ignore=partners) is None for s in (1, -1)):
            swings[which] += 1
            detail[which].append(len(moving))
    for q in reqs:
        if q["kind"] == "spin":
            ok, why = requirement_met(parts, {"kind": "spin", "count": q["count"], "parallel": True})
            fn = verify_function(parts, min_axle_lines=2)
            lowest = not any(i.code == "body_drags" for i in fn.issues)
            out[q["name"]] = (ok and lowest and st.passed, f"{why}; wheels lowest: {lowest}")
    need = {"horizontal": sum(q["count"] for q in reqs if q["kind"] == "swing" and q["axis"] == "horizontal"),
            "vertical": sum(q["count"] for q in reqs if q["kind"] == "swing" and q["axis"] == "vertical")}
    for which in ("horizontal", "vertical"):
        if need[which]:
            out[f"{which}_surfaces"] = (swings[which] >= need[which],
                                        f"{swings[which]} of {need[which]} freed surfaces swing clear "
                                        f"(sizes {detail[which][:8]})")
    return out


def main() -> None:
    import os

    from dotenv import load_dotenv

    from .assembly import FireworksSampler

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("body", type=Path)
    ap.add_argument("--object", required=True)
    ap.add_argument("--requirements", type=Path, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default="base")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=12000)
    args = ap.parse_args()
    original = json.loads(args.body.read_text())
    reqs = json.loads(args.requirements.read_text())
    sampler = FireworksSampler(args.model)
    chat = messages(args.object, original, reqs)
    trace = {"object": args.object, "model": args.model or "base", "requirements": reqs, "rounds": []}
    out_dir = args.body.parent / f"free_edit_{args.label}"
    out_dir.mkdir(exist_ok=True)
    for rnd in range(1, args.rounds + 1):
        entry = {"round": rnd}
        try:
            sample = sampler(chat, max_tokens=args.max_tokens, temperature=0.0 if rnd == 1 else 0.7, timeout=600)
            entry.update(tokens=sample.tokens, answer=sample.text)
            edits = edits_from(sample.text)
            applied = apply_edits(original, edits)
            verdict = check(original, applied.parts, applied.sources, reqs)
            entry.update(edits=len(edits), ops={op: sum(e["op"] == op for e in edits) for op in ("remove", "set", "add")},
                         checks={k: [bool(v[0]), str(v[1])] for k, v in verdict.items()})
            passed = all(v[0] for v in verdict.values())
            (out_dir / f"round{rnd}.json").write_text(json.dumps(applied.parts))
        except Exception as e:  # noqa: BLE001 - parse/apply/service failures are verdicts too
            passed, verdict = False, {"answer": (False, f"{type(e).__name__}: {e}"[:300])}
            entry["checks"] = {k: [False, v[1]] for k, v in verdict.items()}
            sample = locals().get("sample")
        entry["passed"] = passed
        trace["rounds"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "answer"})[:900], flush=True)
        (out_dir / "trace.json").write_text(json.dumps(trace, indent=1))
        if passed:
            break
        issues = [f"{k}: {v[1]}" for k, v in verdict.items() if not v[0]]
        if sample is not None:
            chat = chat + [{"role": "assistant", "content": sample.text},
                           {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))}]


if __name__ == "__main__":
    main()
