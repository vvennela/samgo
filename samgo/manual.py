"""Certified building instructions: an exact, independently checked certificate for every step of the manual.

The manual builds the body upright on the table, one layer per step (parts of one layer never touch each other),
then adds the running gear and other axle-line mechanisms as sub-assemblies. For every step, exactly (rational
arithmetic, no sampling, no floating point trusted):

1. inventory: the manual's parts are exactly the model's parts (multiset equality);
2. insertion: each body part goes straight down without crossing a placed part (the audit's swept-volume rule);
   each axle slides in along its line through Technic holes only, and each wheel, bush or half bush slides on from
   its own side without its circle meeting a placed part;
3. equilibrium: after each step the placed body is in exact force and torque balance under the declared model
   (MODEL below): unit weight per plate cell, four contact points per shared stud, the table pushing up only,
   compression unbounded, stud grip (tension) at most TENSION per stud, the same rule as stability.py;
4. the finished body is one stud-connected assembly.

Witness forces come from Z3 (exact rationals); `check` re-derives every contact and re-checks every witness without
Z3. Hinge, click-hinge and turntable halves count as plates (held still while building), Technic bricks as bricks.
Not modelled: gear weight and gear loads, insertion force, hand access, tolerances; final motion is a sampled check.

    uv run python -m samgo.manual runs/demo/chair/chair.json   # writes <stem>_manual.json beside it
"""

import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

from .certificates import CertificateError, check_path, contacts, digest, rational, xy_overlap
from .parts import parse_piece

VERSION = "samgo-manual-certificate-v1"
TENSION = "250"
MODEL = {"weight": "w*l*height per part, at its centre", "contacts": "4 points per shared stud, 1/4 stud inset",
         "ground": "the table under the lowest layer pushes up only", "compression": "unbounded",
         "tension_per_stud": TENSION, "insertion": "body straight down; axles along their line; wheels/bushes along "
         "the axle", "not_modelled": ["gear weight and loads", "insertion force", "hand access", "tolerances",
                                       "final motion (sampled check elsewhere)"]}
AS_PLATE = {"hinge_base", "hinge_top", "click_a", "click_b", "turntable_base", "turntable_top", "bar_plate", "clip_plate"}
BODY = {"plate", "brick", "technic_brick"} | AS_PLATE
AXIAL = {"axle", "wheel", "bush", "half_bush"}


def split(parts: list[dict]) -> tuple[list[int], list[int], list[int]]:
    """(body part indices, axial gear indices, indices this manual cannot certify)."""
    body = [i for i, p in enumerate(parts) if p["type"] in BODY]
    gear = [i for i, p in enumerate(parts) if p["type"] in AXIAL]
    other = [i for i, p in enumerate(parts) if p["type"] not in BODY | AXIAL]
    return body, gear, other


HEIGHTS = {"plate": 1, "brick": 3, "technic_brick": 3} | {t: 1 for t in AS_PLATE}
PAIRS = (("hinge_base", "hinge_top"), ("click_a", "click_b"))


def units(parts: list[dict], body: list[int]) -> list[list[int]]:
    """Body parts grouped into building units: a hinge's two halves side by side at one height (same pin) are one
    unit, held still while building; every other part is its own unit."""
    used, out = set(), []
    for i in body:
        if i in used:
            continue
        p, mate = parts[i], None
        for a, b in PAIRS:
            if p["type"] in (a, b):
                other = b if p["type"] == a else a
                mate = next((j for j in body if j not in used and j != i and parts[j]["type"] == other
                             and parts[j]["z"] == p["z"] and parts[j].get("pin_x") == p.get("pin_x")
                             and parts[j].get("pin_y") == p.get("pin_y")), None)
        group = [i] + ([mate] if mate is not None else [])
        used.update(group)
        out.append(group)
    return out


def pieces(parts: list[dict], groups: list[list[int]], flipped: bool = False, base: int | None = None):
    """One catalog piece per building unit, in a building frame: upright with layer `base` (default: the lowest)
    on the table, or upside down with the top layer on the table."""
    top = max(parts[i]["z"] + HEIGHTS[parts[i]["type"]] for g in groups for i in g)
    z0 = min(parts[i]["z"] for g in groups for i in g)
    out = []
    for g in groups:
        x0 = min(parts[i]["x"] for i in g); x1 = max(parts[i]["x"] + parts[i]["w"] for i in g)
        y0 = min(parts[i]["y"] for i in g); y1 = max(parts[i]["y"] + parts[i]["l"] for i in g)
        p = parts[g[0]]
        kind = "plate" if p["type"] in AS_PLATE else "brick" if p["type"] == "technic_brick" else p["type"]
        z = top - (p["z"] + HEIGHTS[p["type"]]) if flipped else p["z"] - (z0 if base is None else base)
        out.append(parse_piece({"type": kind, "w": x1 - x0, "l": y1 - y0, "x": x0, "y": y0, "z": z, "color": "black"}))
    return out


def _forces_ok(ps, placed, witness, tension: Fraction) -> None:
    """Exact check of one step's witness (raises CertificateError)."""
    cs = contacts(ps, placed)
    if not isinstance(witness, list) or len(witness) != len(cs):
        raise CertificateError("force witness length differs from the derived contacts")
    force = {i: Fraction(0) for i in placed}
    tx, ty, studs = dict(force), dict(force), {}
    for (up, down, x, y, stud), encoded in zip(cs, witness):
        f = rational(encoded)
        if down is None and f < 0:
            raise CertificateError("the table cannot pull")
        if stud is not None:
            studs[stud] = studs.get(stud, Fraction(0)) + f
        for k, sign in ((up, 1), (down, -1)):
            if k is None:
                continue
            p = ps[k]
            force[k] += sign * f
            tx[k] += sign * f * (x - p.x - Fraction(p.w, 2))
            ty[k] += sign * f * (y - p.y - Fraction(p.l, 2))
    if any(v < -tension for v in studs.values()):
        raise CertificateError("stud grip (tension) above the declared capacity")
    for k in placed:
        p = ps[k]
        if force[k] != p.w * p.l * p.h or tx[k] != 0 or ty[k] != 0:
            raise CertificateError(f"force/torque balance fails exactly for body part {k}")


def _solve(ps, placed, tension: Fraction, deadline: float):
    import z3

    def q(f):
        f = Fraction(f)
        return z3.RealVal(f"{f.numerator}/{f.denominator}")

    cs = contacts(ps, placed)
    s = z3.SolverFor("QF_LRA")
    fs = [z3.Real(f"f{k}") for k in range(len(cs))]
    totals = {k: [[], [], []] for k in placed}
    studs = {}
    for (up, down, x, y, stud), f in zip(cs, fs):
        if down is None:
            s.add(f >= 0)
        if stud is not None:
            studs.setdefault(stud, []).append(f)
        for k, sign in ((up, 1), (down, -1)):
            if k is None:
                continue
            p = ps[k]
            totals[k][0].append(sign * f)
            totals[k][1].append(sign * f * q(x - p.x - Fraction(p.w, 2)))
            totals[k][2].append(sign * f * q(y - p.y - Fraction(p.l, 2)))
    for values in studs.values():
        s.add(z3.Sum(values) >= -q(tension))
    for k, (force, tx, ty) in totals.items():
        p = ps[k]
        s.add(z3.Sum(force) == p.w * p.l * p.h, z3.Sum(tx) == 0, z3.Sum(ty) == 0)
    s.set(timeout=max(1, int((deadline - time.monotonic()) * 1000)))
    r = s.check()
    if r != z3.sat:
        return None, "no equilibrium within the declared stud grip" if r == z3.unsat else "solver: " + s.reason_unknown()
    m = s.model()
    return [str(Fraction(v.numerator_as_long(), v.denominator_as_long()))
            for v in (m.eval(f, model_completion=True) for f in fs)], None


def _connected(ps) -> bool:
    adj = {i: set() for i in range(len(ps))}
    for i, a in enumerate(ps):
        for j, b in enumerate(ps[:i]):
            if xy_overlap(a, b) and (a.z == b.z + b.h or b.z == a.z + a.h):
                adj[i].add(j)
                adj[j].add(i)
    seen, todo = {0}, [0]
    while todo:
        for j in adj[todo.pop()] - seen:
            seen.add(j)
            todo.append(j)
    return len(seen) == len(ps)


def _gear_groups(parts: list[dict], gear: list[int]) -> list[list[int]]:
    """Axle-line sub-assemblies: everything on one line (axis, edge, height), axle first."""
    lines: dict[tuple, list[int]] = {}
    for i in gear:
        p = parts[i]
        lines.setdefault((p.get("axis", "y"), p["x"] if p.get("axis", "y") == "y" else p["y"], p["z"]), []).append(i)
    order = {"axle": 0, "half_bush": 1, "bush": 1, "wheel": 2}
    return [sorted(v, key=lambda i: order[parts[i]["type"]]) for _, v in sorted(lines.items())]


def _gear_path_ok(parts: list[dict], body: list[int], placed_gear: list[int], i: int) -> None:
    """Exact slide-in check for axle-line parts (raises CertificateError). Units: studs and plates as Fractions."""
    from .technic import HOLE_HEIGHT, PLATE, STUD, WHEELS, wheel_studs

    p = parts[i]
    if p.get("axis", "y") == "x":  # the same check with x and y swapped
        swap = lambda q: q | {"x": q["y"], "y": q["x"], "w": q.get("l", q.get("w")), "l": q.get("w", q.get("l")),
                              "axis": "y"} if "w" in q else q | {"x": q["y"], "y": q["x"], "axis": "y"}
        view = [swap(q) for q in parts]
        return _gear_path_ok(view, body, placed_gear, i)
    if p.get("axis", "y") != "y":
        raise CertificateError(f"part {i}: vertical axle lines are not covered")
    edge, zc = Fraction(p["x"]), Fraction(p["z"]) + Fraction(str(HOLE_HEIGHT)) / Fraction(str(PLATE))  # plates
    y0 = Fraction(str(p["y"]))
    span = {"axle": Fraction(p.get("length", 0)), "wheel": Fraction(wheel_studs(p.get("size", "30.4x14"))),
            "bush": Fraction(1), "half_bush": Fraction(1, 2)}[p["type"]]
    if p["type"] == "axle":
        radius = Fraction(0)
    elif p["type"] == "wheel":
        radius = Fraction(str(WHEELS[p.get("size", "30.4x14")][0])) / 2 / Fraction(str(STUD))  # studs
    else:
        radius = Fraction(9, 20)
    hh = {"plate": 1, "brick": 3, "technic_brick": 3} | {t: 1 for t in AS_PLATE}
    to_studs = Fraction(str(PLATE)) / Fraction(str(STUD))
    cz = zc * to_studs                                       # disc centre height, in studs

    def slide(sweep_lo, sweep_hi) -> int | None:
        """The first body part met when sweeping along the axle line over y in (sweep_lo, sweep_hi), else None."""
        for b in body:
            q = parts[b]
            if not (q["y"] < sweep_hi and sweep_lo < q["y"] + q["l"]):
                continue
            if p["type"] == "axle":
                on_line = q["x"] < edge < q["x"] + q["w"] and q["z"] < zc < q["z"] + hh[q["type"]]
                if on_line and not (q["type"] == "technic_brick" and q["z"] == p["z"]):
                    return b
                continue
            dx = max(Fraction(q["x"]) - edge, Fraction(0), edge - (q["x"] + q["w"]))
            dz = max(Fraction(q["z"]) * to_studs - cz, Fraction(0), cz - Fraction(q["z"] + hh[q["type"]]) * to_studs)
            if dx * dx + dz * dz < radius * radius:
                return b
        return None

    def drop(direction: int) -> int | None:
        """The first body part met when the disc comes straight down (+1: from above) or up (-1: from below)."""
        for b in body:
            q = parts[b]
            if not (q["y"] < y0 + span and y0 < q["y"] + q["l"]):
                continue
            if not (q["x"] < edge + radius and edge - radius < q["x"] + q["w"]):
                continue
            qz0, qz1 = Fraction(q["z"]) * to_studs, Fraction(q["z"] + hh[q["type"]]) * to_studs
            if (direction > 0 and qz1 > cz - radius) or (direction < 0 and qz0 < cz + radius):
                return b
        return None

    if p["type"] == "axle":  # an axle goes in from one end, through Technic holes only
        hit = [slide(Fraction(-10**6), y0 + span), slide(y0, Fraction(10**6))]
    else:                    # a wheel or bush slides on from either end, or drops into a gap from above or below
        hit = [slide(Fraction(-10**6), y0 + span), slide(y0, Fraction(10**6)), drop(1), drop(-1)]
    if all(h is not None for h in hit):
        raise CertificateError(f"{p['type']} part {i} has no clear way in (blocked by part {hit[0]})")


MAX_STEP = 6  # parts per step, as in LEGO's own manuals


def _feasible(ps, placed, tension: Fraction) -> bool:
    """Fast floating-point screen (HiGHS) for the same equilibrium; only steps that pass get an exact Z3 witness,
    and only exact witnesses enter the certificate."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    cs = contacts(ps, placed)
    if not cs:
        return False
    row = {k: n for n, k in enumerate(placed)}
    a_eq = lil_matrix((3 * len(placed), len(cs)))
    b_eq = np.zeros(3 * len(placed))
    for n, (up, down, x, y, _) in enumerate(cs):
        for k, sign in ((up, 1), (down, -1)):
            if k is None:
                continue
            p, r = ps[k], 3 * row[k]
            a_eq[r, n] += sign
            a_eq[r + 1, n] += sign * float(x - p.x - Fraction(p.w, 2))
            a_eq[r + 2, n] += sign * float(y - p.y - Fraction(p.l, 2))
    for k in placed:
        b_eq[3 * row[k]] = ps[k].w * ps[k].l * ps[k].h
    studs = {}
    for n, (_, _, _, _, stud) in enumerate(cs):
        if stud is not None:
            studs.setdefault(stud, []).append(n)
    a_ub = lil_matrix((len(studs), len(cs)))
    for r, ns in enumerate(studs.values()):
        for n in ns:
            a_ub[r, n] = -1.0                                  # -(sum) <= tension
    bounds = [(0, None) if down is None else (None, None) for (_, down, _, _, _) in cs]
    res = linprog(np.zeros(len(cs)), A_ub=a_ub.tocsr() if studs else None,
                  b_ub=np.full(len(studs), float(tension) * 0.999) if studs else None,
                  A_eq=a_eq.tocsr(), b_eq=b_eq, bounds=bounds, method="highs")
    return res.status == 0


def frame(parts, groups, spec):
    """Pieces of a building frame and the unit -> piece index map. spec: {"subset": units on/over the table,
    "flipped": upside down, "base": the layer standing on the table (upright; default the lowest)}."""
    sub = spec["subset"]
    ps = pieces(parts, [groups[k] for k in sub], spec.get("flipped", False), spec.get("base"))
    return ps, {k: n for n, k in enumerate(sub)}


def _closure_steps(parts, groups, phases, tension, deadline, assembly="main", placed=None):
    """Steps of at most MAX_STEP units, each leaving the placed assembly in exact balance with no temporary
    support. A unit that cannot stand alone goes in together with what locks it: the units directly on top of it
    (caps) and the caps' own unplaced supports. `phases`: [(frame spec, units to add)]."""
    placed, steps = list(placed or []), []
    for spec, allowed in phases:
        ps, local = frame(parts, groups, spec)
        loc = lambda ks: [local[k] for k in ks]
        on = {k: [j for j in allowed if ps[local[j]].z == ps[local[k]].z + ps[local[k]].h
                  and xy_overlap(ps[local[j]], ps[local[k]])] for k in allowed}
        under = {k: [j for j in allowed if ps[local[k]].z == ps[local[j]].z + ps[local[j]].h
                     and xy_overlap(ps[local[j]], ps[local[k]])] for k in allowed}
        remaining = set(allowed)
        P = lambda k: ps[local[k]]
        while remaining:
            if time.monotonic() > deadline:
                return None, "time budget exhausted"
            accepted = None
            for k in sorted(remaining, key=lambda k: (P(k).z, P(k).x, P(k).y)):
                group = [k]
                while len(group) <= MAX_STEP and accepted is None:
                    if not any(j not in group and P(j).z < P(g).z and xy_overlap(P(j), P(g))
                               for g in group for j in remaining):  # never cover a unit still waiting
                        order = sorted(group, key=lambda g: (P(g).z, P(g).x, P(g).y))
                        try:
                            trial = list(placed)
                            for g in order:
                                check_path(ps, loc(trial), local[g])
                                trial.append(g)
                            if _feasible(ps, loc(trial), tension):
                                witness, _ = _solve(ps, loc(trial), tension, deadline)
                                if witness is not None:
                                    accepted = (order, witness)
                        except CertificateError:
                            pass
                    if accepted is not None:
                        break
                    caps = {j for g in group for j in on[g] if j in remaining} - set(group)
                    if not caps:
                        break
                    grown = set(group) | caps | {j for c in caps for j in under[c] if j in remaining}
                    if len(grown) > MAX_STEP or grown == set(group):
                        break
                    group = sorted(grown)
                if accepted is not None:
                    break
            if accepted is None:
                low = sorted(remaining, key=lambda k: (P(k).z, P(k).x, P(k).y))[:3]
                why = []
                for k in low:
                    try:
                        check_path(ps, loc(placed), local[k])
                        reason = "path clear, but it cannot stand alone or with its caps"
                    except CertificateError as e:
                        reason = str(e)
                    p = parts[groups[k][0]]
                    why.append(f"{p['type']} {p.get('w')}x{p.get('l')} at ({p['x']},{p['y']},{p['z']}): {reason}")
                return None, (f"stuck after {len(placed)} units: no step of up to {MAX_STEP} parts stands on its own; "
                              "lowest waiting: " + "; ".join(why))
            order, witness = accepted
            placed += order
            remaining -= set(order)
            steps.append({"kind": "layer", "assembly": assembly, "frame": spec, "units": order,
                          "parts": [i for k in order for i in groups[k]], "forces": witness})
    return _merged(steps), None


def _merged(steps):
    """Consecutive small steps in one frame joined up to MAX_STEP parts: the joined step ends in the last one's
    certified state, and its insertion order is kept, so the certificate is unchanged in substance."""
    out = []
    for st in steps:
        last = out[-1] if out else None
        if last and last["frame"] == st["frame"] and last.get("assembly") == st.get("assembly") \
                and len(last["parts"]) + len(st["parts"]) <= MAX_STEP:
            last.update(units=last["units"] + st["units"], parts=last["parts"] + st["parts"], forces=st["forces"])
        else:
            out.append(dict(st))
    return out


def _with_module(parts, groups, z, levels, area, tension, deadline, hint=None):
    """Lower body built from its largest low layer (then turned over for what lies below it), an upper
    sub-assembly (everything from a cut layer up) built on its own upside down, then the lower body set upright and
    the module set on in one step. Tries cut layers from the top down. (steps, name, None) or (None, None, why)."""
    everything = list(range(len(groups)))
    last = "no cut layer tried"
    cuts = levels[2:]
    if hint is not None:  # start at the height where whole-model orders got stuck: the module begins there
        cuts = sorted(cuts, key=lambda c: (abs(c - hint), -c))
    else:
        cuts = sorted(cuts, reverse=True)
    for cut in cuts:
        lower = [k for k in everything if z[k] < cut]
        upper = [k for k in everything if z[k] >= cut]
        lower_levels = sorted({z[k] for k in lower})
        b = max(lower_levels[1:] or lower_levels, key=lambda lv: area[lv])
        low_phases = [({"subset": lower, "flipped": False, "base": b}, [k for k in lower if z[k] >= b]),
                      ({"subset": lower, "flipped": True, "base": None}, [k for k in lower if z[k] < b])]
        low, issue = _closure_steps(parts, groups, low_phases, tension, deadline)
        if low is None:
            last = f"cut z={cut}: lower body: {issue}"
            continue
        mod, issue = _closure_steps(parts, groups, [({"subset": upper, "flipped": True, "base": None}, upper)],
                                    tension, deadline, assembly="module")
        if mod is None:
            last = f"cut z={cut}: module: {issue}"
            continue
        final = {"subset": everything, "flipped": False, "base": None}
        ps, local = frame(parts, groups, final)
        try:
            for k in upper:  # the whole module comes straight down onto the lower body
                check_path(ps, [local[j] for j in lower], local[k])
        except CertificateError as e:
            last = f"cut z={cut}: setting the module on: {e}"
            continue
        upright, _ = _solve(ps, [local[j] for j in lower], tension, deadline)  # lower body turned upright, alone
        attached, _ = _solve(ps, [local[j] for j in everything], tension, deadline)
        if upright is None or attached is None:
            last = f"cut z={cut}: {'the lower body cannot stand upright' if upright is None else 'the joined model cannot stand'}"
            continue
        steps = low + mod + [{"kind": "layer", "assembly": "main", "frame": final, "units": [], "parts": [],
                              "forces": upright},
                             {"kind": "attach", "assembly": "main", "frame": final, "units": upper,
                              "parts": [i for k in upper for i in groups[k]], "forces": attached}]
        return steps, f"lower body from layer z={b}; upper sub-assembly from layer z={cut} built upside down, set on", None
    return None, None, f"with a sub-assembly: {last}"


def certify_manual(parts: list[dict], *, timeout: float | None = None) -> dict:
    """The manual (steps of original part indices) and its certificate, or why it could not be certified."""
    start = time.monotonic()
    timeout = timeout or (600 + 6 * len(parts))  # bigger models get more search time
    body, gear, other = split(parts)
    report = {"version": VERSION, "model": MODEL, "build_sha256": digest(parts), "passed": False, "steps": [],
              "uncertified_parts": [{"index": i, "type": parts[i]["type"]} for i in other], "issues": []}
    if not body:
        report["issues"].append("no body parts")
        return report
    groups = units(parts, body)
    tension = rational(TENSION)
    failures = []
    # How manuals are made: steps of up to MAX_STEP parts, each locking what it adds; built upright, upside down,
    # from a layer out (then turned over), or with a sub-assembly built on its own and set on in one step.
    everything = list(range(len(groups)))
    z = {k: parts[groups[k][0]]["z"] for k in everything}
    levels = sorted(set(z.values()))
    area = {lv: sum(parts[groups[k][0]]["w"] * parts[groups[k][0]]["l"] for k in everything if z[k] == lv)
            for lv in levels}
    allspec = lambda flipped=False, base=None: {"subset": everything, "flipped": flipped, "base": base}
    options = [("upright", [(allspec(), everything)]), ("upside down", [(allspec(True), everything)])]
    for b in sorted(levels[1:], key=lambda lv: -area[lv])[:4]:
        options.append((f"from layer z={b} up, then turned over",
                        [(allspec(False, b), [k for k in everything if z[k] >= b]),
                         (allspec(True), [k for k in everything if z[k] < b])]))
    for name, phases in options:
        steps, issue = _closure_steps(parts, groups, phases, tension, start + timeout)
        if steps is not None:
            report.update(orientation=name, steps=steps)
            break
        failures.append(f"{name}: {issue}")
    if not report["steps"]:
        # The attempt that got furthest shows where a sub-assembly has to begin.
        stuck = [(int(m.group(1)), int(m.group(2))) for f in failures
                 for m in [re.search(r"stuck after (\d+) units.*?lowest waiting: \S+ \S+ at \(\d+,\d+,(\d+)\)", f)] if m]
        steps, name, issue = _with_module(parts, groups, z, levels, area, tension, start + timeout,
                                          hint=max(stuck)[1] if stuck else None)
        if steps is not None:
            report.update(orientation=name, steps=steps)
        else:
            failures.append(issue)
    if not report["steps"]:
        report["issues"] += failures
        report["seconds"] = time.monotonic() - start
        return report
    try:
        done: list[int] = []
        for grp in _gear_groups(parts, gear):
            for i in grp:
                _gear_path_ok(parts, body, done, i)
                done.append(i)
            report["steps"].append({"kind": "axle_line", "parts": grp})
    except CertificateError as e:
        report["issues"].append(str(e))
        report["seconds"] = time.monotonic() - start
        return report
    checked = check(parts, report)
    report["passed"] = checked["passed"] and not other
    if not checked["passed"]:
        report["issues"] += checked["issues"]
    if other:
        report["issues"].append(f"{len(other)} parts of types this manual cannot certify (e.g. pins, liftarms)")
    report["seconds"] = time.monotonic() - start
    return report


def check(parts: list[dict], report: dict) -> dict:
    """Independent re-check of a manual certificate (no solver): inventory, insertion, every witness."""
    try:
        if report.get("version") != VERSION or report.get("build_sha256") != digest(parts):
            raise CertificateError("certificate does not belong to this model")
        body, gear, other = split(parts)
        listed = [i for s in report["steps"] for i in s["parts"]]
        if sorted(listed) != sorted(body + gear):
            raise CertificateError("the manual's parts differ from the model's parts")
        groups = units(parts, body)
        placed = {"main": [], "module": []}
        done: list[int] = []
        tension = rational(TENSION)
        for s in report["steps"]:
            if s["kind"] in ("layer", "attach"):
                if sorted(i for k in s["units"] for i in groups[k]) != sorted(s["parts"]):
                    raise CertificateError("a step's units and parts disagree")
                ps, local = frame(parts, groups, s["frame"])
                mine = placed[s.get("assembly", "main")]
                if s["kind"] == "attach":  # a finished module, set on as one piece
                    if sorted(s["units"]) != sorted(placed["module"]):
                        raise CertificateError("the attached module is not the module that was built")
                    for k in s["units"]:
                        check_path(ps, [local[j] for j in mine], local[k])
                    mine.extend(s["units"])
                else:
                    for k in s["units"]:
                        check_path(ps, [local[j] for j in mine], local[k])
                        mine.append(k)
                _forces_ok(ps, [local[j] for j in mine], s["forces"], tension)
            else:
                for i in s["parts"]:
                    _gear_path_ok(parts, body, done, i)
                    done.append(i)
        if sorted(placed["main"]) != list(range(len(groups))):
            raise CertificateError("not every body part ends up in the finished model")
        ps, _ = frame(parts, groups, {"subset": list(range(len(groups))), "flipped": False, "base": None})
        if not _connected(ps):
            raise CertificateError("the finished body is not one stud-connected assembly")
        return {"passed": True, "steps": len(report["steps"])}
    except (CertificateError, KeyError, TypeError, ValueError) as e:
        return {"passed": False, "issues": [str(e)]}


def main() -> None:
    for path in map(Path, sys.argv[1:]):
        parts = json.loads(path.read_text())
        r = certify_manual(parts)
        out = path.with_name(path.stem + "_manual.json")
        out.write_text(json.dumps(r))
        print(f"{path.name}: {'CERTIFIED' if r['passed'] else 'not certified'}; {len(r['steps'])} steps; "
              f"{len(r['uncertified_parts'])} uncertified parts; {r.get('seconds', 0):.0f}s; issues: {r['issues'][:2]}")


if __name__ == "__main__":
    main()
