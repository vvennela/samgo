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
         '{"op":"remove","index":I}, {"op":"set","index":I,"part":{...}}, {"op":"add","part":{...}}, '
         '{"op":"clear","box":[x0,x1,y0,y1,z0,z1]} (empties that inclusive box in the whole model, shown or not: '
         "parts inside go, parts crossing its edge are cut there and the rest re-laid as real plates, crumbs left "
         "hanging fall away; to free a panel, clear a channel on every side but its hinge side). "
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


def check(original: list[dict], parts: list[dict], sources: list[int | None], reqs: list[dict],
          stand: bool = True, crop: bool = False) -> dict:
    """Generic verdicts. `sources[i]`: the original index of candidate part i (None if added). `crop`: the parts are
    a neighbourhood cut out of a bigger model (it need not reach the table, and may be in pieces that join outside
    it): connected then means the edit made no new separate piece and left no joint half holding nothing."""
    from .kinematics import _sides, analyse, connected, locked, requirement_met, sweep
    from .mechanism_bases import PIVOT_AS, structural
    from .stability import stability
    from .technic import structural as technic_structural, verify_function
    from .verifier import verify

    out = {}
    r = verify(technic_structural(structural(parts)), None)
    rule = [i for i in r.issues if i.code not in ("floating", "disconnected")]
    from .animation import through_solid

    pierced = through_solid(parts)
    msgs = [i.message for i in rule[:4]] + [
        f"the {parts[a]['type']} at ({parts[a]['x']}, {parts[a]['y']}, {parts[a]['z']}) passes through the solid "
        f"{parts[b]['type']} at ({parts[b]['x']}, {parts[b]['y']}, {parts[b]['z']}) (not through a hole)"
        for a, b in pierced[:4]]
    out["structure"] = (not rule and not pierced, "; ".join(msgs) or "valid parts, no overlaps")
    if crop:
        from .kinematics import dangling, groups

        ok = groups(parts) <= groups(original) and not dangling(parts)
        out["connected"] = (ok, "no new loose piece" if ok else "the edit left a piece or joint half unattached")
    else:
        out["connected"] = (connected(parts), "one structure through studs and joints")
    if stand:
        st = stability(locked(parts, PIVOT_AS), vehicle=any(p["type"] == "wheel" for p in parts))
    else:  # a part-level task: the whole object's balance is checked when every mechanism is in
        from types import SimpleNamespace
        st = SimpleNamespace(passed=True, issues=[])
    out["stands"] = (st.passed, "; ".join(i.message for i in st.issues) or "stands")
    swings = {"horizontal": 0, "vertical": 0}
    detail = {"horizontal": [], "vertical": []}
    need = {"horizontal": sum(q["count"] for q in reqs if q["kind"] == "swing" and q["axis"] == "horizontal"),
            "vertical": sum(q["count"] for q in reqs if q["kind"] == "swing" and q["axis"] == "vertical")}
    if any(need.values()):  # only asked-for swings are swept: smallest moving sets first, stop once enough swing
        bodies, joints = analyse(parts)
        candidates = []
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
            if need[which]:
                candidates.append((len(moving), k, moving, point, axis, partners, which))
        for _, _, moving, point, axis, partners, which in sorted(candidates, key=lambda c: c[:2]):
            if all(swings[w] >= need[w] for w in need):
                break
            if swings[which] >= need[which]:
                continue
            wanted = [q["degrees"] for q in reqs if q["kind"] == "swing" and q["axis"] == which]
            degrees = min(wanted) if wanted else 20
            if any(sweep(parts, moving, point, axis, s * degrees, ignore=partners) is None for s in (1, -1)):
                swings[which] += 1
                detail[which].append(len(moving))
    for q in reqs:
        if q["kind"] == "rotor":  # something on an axle turns all the way round, clear of everything
            from .kinematics import rotors

            n, why = rotors(parts, q.get("axis", "horizontal"))
            out[q["name"]] = (n >= q.get("count", 1), why)
        if q["kind"] == "spin":
            ok, why = requirement_met(parts, {"kind": "spin", "count": q["count"], "parallel": True})
            fn = verify_function(parts, min_axle_lines=2)
            lowest = not any(i.code == "body_drags" for i in fn.issues)
            stands = st.passed or not q.get("stands", True)  # one leg of several is not asked to stand alone
            out[q["name"]] = (ok and lowest and stands, f"{why}; wheels lowest: {lowest}")
    for which in ("horizontal", "vertical"):
        if need[which]:
            out[f"{which}_surfaces"] = (swings[which] >= need[which],
                                        f"{swings[which]} of {need[which]} freed surfaces swing clear "
                                        f"(sizes {detail[which][:8]})")
    return out


def _rotor_feedback(parts: list[dict], name) -> list[str]:
    """Why nothing turns all the way round: no axle in a round hole yet, the axle is locked, or what the turning
    parts hit (by name), for any object."""
    import numpy as np

    from .kinematics import BEAMS, HEIGHT as HGT, STUD, _corners, _inside, _sides, analyse

    bodies, joints = analyse(parts)
    axles = [k for k, j in enumerate(joints) if j[2] == "revolute" and abs(j[4][2]) < 0.1
             and any(parts[i].get("type") == "axle" for x in (j[0], j[1]) for i in bodies[x])]
    if not axles:
        return ["rotor: nothing turns on a horizontal axle yet: set a Technic brick into the face (clear its slot "
                "first), run an axle through its round hole out of the face, and lock arms on the axle outside the "
                "face (a 2-long liftarm with an axle hole), with bushes holding the axle"]
    msgs = []
    for k in axles[:2]:
        a, b, kind, point, axis, partners = joints[k]
        where = f"the axle at (x={point[0] / STUD:.1f}, y={point[1] / STUD:.1f}) studs"
        sides = _sides(bodies, joints, k)
        if sides is None:
            msgs.append(f"rotor: {where} cannot turn: what is on it is also held by studs or a second axle hole "
                        "elsewhere; the arms must touch only the axle")
            continue
        small = min(sides, key=lambda g: sum(len(bodies[x]) for x in g))
        moving = [i for x in small for i in bodies[x]]
        if not any(parts[i].get("type") in HGT.keys() | BEAMS for i in moving):
            msgs.append(f"rotor: {where} turns, but nothing is locked on it: add arms with an axle hole "
                        "(liftarm length 2, axle_hole start or end) on the axle outside the face")
            continue
        fixed = [n for n in range(len(parts)) if n not in moving and n not in partners
                 and parts[n].get("type") in HGT]
        pts = np.vstack([_corners(parts[i]) for i in moving if parts[i].get("type") in HGT.keys() | BEAMS]
                        or [np.zeros((0, 3))])
        if not len(pts):
            msgs.append(f"rotor: the parts on {where} hit something when they turn; move the arms further out "
                        "along the axle, away from the face")
            continue
        ax = np.asarray(axis, float) / np.linalg.norm(axis)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        hits = set()
        for deg in range(15, 360, 15):
            th = np.radians(deg)
            R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
            hits |= {n for n in fixed if _inside((pts - point) @ R.T + point, parts[n])}
        msgs.append(f"rotor: the parts on {where} hit " + "; ".join(name(n) for n in sorted(hits)[:3])
                    + " when they turn: move the arms further out along the axle or clear what they hit"
                    if hits else f"rotor: the parts on {where} hit something (the ground or the face) when they "
                    "turn: move the arms further out along the axle, or put the axle higher")
    return msgs


def explain(original: list[dict], parts: list[dict], sources: list[int | None], changed, verdict: dict,
            local: list[int] | None = None) -> list[str]:
    """Say what is wrong in terms the model can act on, for any object: which of its parts overlap which numbered
    input part (and that the input part must be removed first), which parts rest on nothing, what a hinge's swing
    runs into, why no freed surface turns, what sits below the wheels."""
    import numpy as np

    from .kinematics import HEIGHT as HGT, STUD, box, _corners, _inside, _sides, analyse
    from .running_gear_task import describe

    local_of = {i: k for k, i in enumerate(local if local is not None else range(len(original)))}

    def name(n):
        s = sources[n]
        k = local_of.get(s) if s is not None else None
        label = ("your added" if s is None else f"your changed part {k}" if n in changed
                 else f"input part {k}" if k is not None else "a body part you were not shown")
        return f"{label} ({describe(parts[n])})"

    msgs = []
    owners: dict = {}
    for n, p in enumerate(parts):
        if p.get("type") in HGT and "w" in p and "l" in p:
            for x in range(p["x"], p["x"] + p["w"]):
                for y in range(p["y"], p["y"] + p["l"]):
                    for z in range(p["z"], p["z"] + HGT[p["type"]]):
                        owners.setdefault((x, y, z), []).append(n)
    pairs = sorted({(o[a], o[b]) for o in owners.values() if len(o) > 1
                    for a in range(len(o)) for b in range(a + 1, len(o))})
    for a, b in pairs[:6]:
        old = [n for n in (a, b) if sources[n] is not None and n not in changed]
        hint = (f": remove {name(old[0]).split(' (')[0].replace('input ', '')} first (add a remove edit for it)"
                if old and len(old) == 1 else ": move one of them")
        msgs.append(f"overlap: {name(a)} and {name(b)} fill the same cells{hint}")
    if len(pairs) > 6:
        msgs.append(f"overlap: {len(pairs) - 6} more overlapping pair(s) of the same kind")
    for key, (ok, why) in verdict.items():
        if ok or (key == "structure" and pairs):
            continue
        if key == "rotor":
            if "turntable" in why:
                msgs.append(f"rotor: {why}")
            elif "below the table" in why or "nothing is locked" in why:
                msgs.append(f"rotor: {why}" + ("" if "below" in why else ": lock arms on it with an axle hole "
                                                "(liftarm length 2, axle_hole start or end); round holes spin loose"))
            else:
                msgs += _rotor_feedback(parts, name)
            continue
        if key.endswith("_surfaces"):
            which = key.split("_")[0]
            bodies, joints = analyse(parts)
            hinges = [k for k, j in enumerate(joints)
                      if ("vertical" if abs(j[4][2]) > 0.9 else "horizontal") == which]
            if not hinges:
                msgs.append(f"{key}: there is no hinge with a {which} axis yet: put a bar plate on the body and a clip "
                            "plate on the surface, at one height, facing each other with one empty stud between")
            for k in hinges[:3]:
                a, b, kind, point, axis, partners = joints[k]
                sides = _sides(bodies, joints, k)
                where = f"the hinge at (x={point[0] / STUD:.1f}, y={point[1] / STUD:.1f}) studs"
                if sides is None:
                    msgs.append(f"{key}: {where} cannot turn: the surface is still attached to the body by studs "
                                "elsewhere; free it with a channel on every side but the hinge side (clear boxes) so the hinge is "
                                "its only link")
                    continue
                small = min(sides, key=lambda g: sum(len(bodies[x]) for x in g))
                moving = [i for x in small for i in bodies[x]]
                if not any(sources[i] is not None for i in moving):
                    msgs.append(f"{key}: {where} turns only parts you added; the surface must be parts of the input "
                                "model, freed from it and hung on the hinge")
                    continue
                fixed = [n for n in range(len(parts)) if n not in moving and n not in partners
                         and parts[n].get("type") in HGT]
                pts = np.vstack([_corners(parts[i]) for i in moving if parts[i].get("type") in HGT] or [np.zeros((0, 3))])
                ax = np.asarray(axis, float) / np.linalg.norm(axis)
                K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                hits = set()
                for deg in (20, -20):
                    for s in range(1, 5):
                        th = np.radians(deg * s / 4)
                        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
                        moved = (pts - point) @ R.T + point
                        hits |= {n for n in fixed if len(pts) and _inside(moved, parts[n])}
                if hits:
                    msgs.append(f"{key}: the {len(moving)} part(s) on {where} hit "
                                + "; ".join(name(n) for n in sorted(hits)[:3])
                                + " when they swing: remove or shorten those to leave a gap")
            continue
        if key == "connected":
            from .kinematics import dangling

            held = dangling(parts)
            if held:
                msgs.append("connected: these hinge parts hold nothing - their studs grip no other part, so they "
                            "only hang on their partner: " + "; ".join(name(n) for n in held[:4])
                            + " - put each on (or under) a part of the side it carries")
            bodies, joints = analyse(parts)
            link: dict = {}
            for j in joints:
                link.setdefault(j[0], set()).add(j[1])
                link.setdefault(j[1], set()).add(j[0])
            groups, seen = [], set()
            for b0 in bodies:
                if b0 in seen:
                    continue
                todo, g = [b0], []
                seen.add(b0)
                while todo:
                    b = todo.pop()
                    g += bodies[b]
                    for c in link.get(b, ()):
                        if c not in seen:
                            seen.add(c)
                            todo.append(c)
                groups.append(g)
            groups.sort(key=len)
            loose = [n for g in groups[:-1] for n in g]
            if loose or not held:
                msgs.append("connected: these parts are not attached to the rest of the model: "
                            + "; ".join(name(n) for n in loose[:4])
                            + " - each must share studs with a part that stays, or sit on a joint")
            continue
        if key != "structure":
            msgs.append(f"{key}: {why}")
            if "lowest: False" in why:
                wheels = [p for p in parts if p.get("type") == "wheel"]
                if wheels:
                    low = min(box(p)[0][2] for p in wheels)
                    under = sorted((n for n, p in enumerate(parts) if p.get("type") in HGT and "w" in p
                                    and box(p)[0][2] < low + 3), key=lambda n: box(parts[n])[0][2])[:3]
                    if under:
                        msgs.append("  parts as low as the wheels: " + "; ".join(name(n) for n in under)
                                    + " - the leg must be long enough that the wheels hang below everything")
        else:
            msgs.append(f"structure: {why}")
    return msgs[:12]


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
