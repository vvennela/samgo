"""Legs with wheels for any object Spark says stands on them - Spark knows where, Glimmer builds, code verifies.

1. Spark, from the photo and a top view of the model (numbered stud ticks), lists the legs or wheel sets the real
   object stands on and where they are on the top view: {"legs": [{"name", "x", "y"}]}. Its own knowledge, no hints.
2. Code raises the model so wheels fit under its lowest point, and for each leg finds the nearest spot on the
   underside where a plain answer (a brick column down from the underside to two Technic bricks, an axle, two wheels,
   two bushes) passes the generic checks - a buildable handoff, never built by code.
3. Glimmer builds each leg from the local parts and the training wording (samgo.primitives.leg_text, L2: "add a leg
   around here"), with the checker's precise feedback between rounds. A leg Glimmer cannot build is left out.
4. Code checks the finished model stands on its wheels (samgo.stability, joints held).

Nothing here knows the object: the same code runs on a chair, a cart or an airliner.
"""

import hashlib
import json
import re
from pathlib import Path

LEG = [{"name": "leg", "kind": "spin", "count": 2, "stands": False}]
CLEAR = 13  # a 30.4 mm wheel on a Technic brick at z = 3 reaches ~9.5 plates: the model's lowest cell sits above it

PROMPT = """The first image is a photo of {obj}. The second is a top view of a LEGO-scale model of it: x runs across
(numbered ticks, one per stud), y runs up (numbered ticks), lighter is higher.

From what you know about the real object: does it stand or roll on legs or wheel sets (a stand, gear, casters,
wheels)? If so, list each one and where it is under the object, read on the top view. Answer with one JSON object,
nothing else:
{{"legs": [{{"name": "<name>", "x": <tick>, "y": <tick>}}]}}
An empty list if the real object has none."""


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group()) if m else None
        return {"legs": [{"name": str(g.get("name", "leg")), "x": int(g["x"]), "y": int(g["y"])}
                         for g in d["legs"]]}
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


async def ask(photo: Path, body: list[dict], work: Path, *, obj: str, client=None) -> dict:
    """Spark's legs (cached per photo, model and prompt)."""
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    from .function_spec import BASE_URL, MODEL, _data_url
    from .part_layout import render_top

    work.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(Path(photo).read_bytes() + json.dumps([PROMPT, body, obj]).encode()).hexdigest()[:16]
    cache = Path("runs/research_cache") / f"legs_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    view = work / "top_view.png"
    render_top(body, set(), view)
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    import os

    client = client or AsyncOpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=BASE_URL, max_retries=6)
    text = PROMPT.format(obj=f"a{'n' if obj[:1] in 'aeiou' else ''} {obj}")
    out = {"error": "no answer"}
    for _ in range(3):
        r = await client.chat.completions.create(
            model=MODEL, max_tokens=8000, extra_body={"reasoning_effort": "low"},
            messages=[{"role": "user", "content": [{"type": "text", "text": text},
                                                   {"type": "image_url", "image_url": {"url": _data_url(photo)}},
                                                   {"type": "image_url",
                                                    "image_url": {"url": _data_url(view, 1400)}}]}])
        answer = r.choices[0].message.content or ""
        out = parse(answer) or {"error": f"unparseable answer: {answer[:200]}"}
        if "error" not in out:
            break
    if "error" not in out:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out


def raised(parts: list[dict]) -> tuple[list[dict], int]:
    """The model moved up so its lowest cell is at CLEAR (wheels fit under it); returns (parts, lift)."""
    lift = CLEAR - min(p["z"] for p in parts)
    return [p | {"z": p["z"] + lift} for p in parts], lift


def reference(parts: list[dict], x0: int, yc: int, bottom: int, color: str = "dark_gray") -> list[dict]:
    """A plain leg under the 2x2 spot x0..x0+1, yc-1..yc whose underside is at `bottom` (the leg primitive's answer,
    the column as tall as the spot needs): bricks, then plates, from z = 6 up to the underside."""
    adds, z = [], 6
    while z + 3 <= bottom:
        adds.append(dict(type="brick", w=2, l=2, x=x0, y=yc - 1, z=z, color=color))
        z += 3
    while z < bottom:
        adds.append(dict(type="plate", w=2, l=2, x=x0, y=yc - 1, z=z, color=color))
        z += 1
    xg = x0 + 1
    adds += [dict(type="technic_brick", w=2, l=1, x=x0, y=y, z=3, color=color, hole="round") for y in (yc - 1, yc)]
    adds += [dict(type="axle", axis="y", x=xg, z=3, y=yc - 4, length=8),
             dict(type="wheel", axis="y", x=xg, z=3, y=yc - 3, size="30.4x14", color="black"),
             dict(type="wheel", axis="y", x=xg, z=3, y=yc + 1, size="30.4x14", color="black"),
             dict(type="bush", axis="y", x=xg, z=3, y=yc - 4), dict(type="bush", axis="y", x=xg, z=3, y=yc + 3)]
    return [{"op": "add", "part": a} for a in adds]


def spots(parts: list[dict]) -> dict[tuple[int, int], int]:
    """2x2 underside spots (x0, yc) -> underside height: the four columns' lowest cells are level, nothing below
    them, and the wheels' sweep (x0-1..x0+2, yc-4..yc+3, z < underside) is empty."""
    from .primitives import occupancy

    occ = occupancy(parts)
    bottom: dict = {}
    for (x, y, z) in occ:
        bottom[(x, y)] = min(bottom.get((x, y), 999), z)
    out = {}
    for (x, y), z in bottom.items():
        cols = [(x + dx, y + dy) for dx in (0, 1) for dy in (-1, 0)]
        if all(bottom.get(c) == z for c in cols) and z >= CLEAR and not any(
                bottom.get((x + dx, y + dy), 999) < z for dx in range(-1, 3) for dy in range(-4, 4)):
            out[(x, y)] = z
    return out


def site(parts: list[dict], x: int, y: int, taken: list[tuple[int, int]], limit: int = 40) -> dict | None:
    """The nearest underside spot to (x, y), clear of legs already placed, where the plain leg passes the checks."""
    from .edits import apply_edits
    from .free_edit_test import check

    found = spots(parts)
    order = sorted(found, key=lambda s: (s[0] + 0.5 - x) ** 2 + (s[1] - 0.5 - y) ** 2)
    order = [s for s in order if all(abs(s[0] - t[0]) > 4 or abs(s[1] - t[1]) > 9 for t in taken)]
    for x0, yc in order[:limit]:
        a = apply_edits(parts, reference(parts, x0, yc, found[(x0, yc)]))
        v = check(parts, a.parts, a.sources, LEG, stand=False)
        if all(v[k][0] for k in ("structure", "connected", "leg")):
            return {"x0": x0, "yc": yc, "bottom": found[(x0, yc)]}
    return None


def local(parts: list[dict], s: dict) -> list[int]:
    return [i for i, p in enumerate(parts) if p["z"] <= s["bottom"] + 3 and abs(p["x"] - s["x0"]) <= 8
            and abs(p["y"] - s["yc"]) <= 8]


def text(s: dict) -> str:
    """The training's L2 wording (samgo.primitives.leg_text), in the model's own axes."""
    return (f"These are the underside parts of a raised object around x = {s['x0']}..{s['x0'] + 1}, y = "
            f"{s['yc'] - 1}..{s['yc']}. Add a leg there: a column down from the underside, an axle and two "
            "wheels that spin, lowest of all.")


def build(photo: Path, parts: list[dict], obj: str, sampler, work: Path, loop, rounds: int = 3,
          max_legs: int = 4) -> tuple[list[dict], dict]:
    """Raise the model and add the legs Spark names that Glimmer can build. Returns (parts, report)."""
    from .spark_mechanisms import attempt

    sp = loop.run_until_complete(ask(photo, parts, work, obj=obj))
    report = {"spark": sp, "legs": []}
    if "legs" not in sp or not sp["legs"]:
        return parts, report
    unraised = parts
    parts, lift = raised(parts)
    report["lift"] = lift
    taken = []
    for g in sp["legs"][:max_legs]:
        entry = dict(g)
        report["legs"].append(entry)
        s = site(parts, g["x"], g["y"], taken)
        if s is None:
            entry["error"] = "no buildable spot under that point"
            continue
        taken.append((s["x0"], s["yc"]))
        instruction = text(s)
        rec = attempt(parts, local(parts, s), instruction, sampler, rounds=rounds, reqs=LEG)
        entry.update(site=s, instruction=instruction, passed=rec["passed"], rounds=rec["rounds"])
        if rec["passed"]:
            parts = rec["parts"]
    if not any(e.get("passed") for e in report["legs"]):  # nothing built: the model stays where it was
        report.pop("lift")
        return unraised, report
    report["stands"] = stands(parts)
    if not report["stands"]["passed"]:  # legs it falls over on (some not built) are not kept: it rests on its base
        report.update(kept=False, stands=stands(unraised), why="the model does not stand on the legs that were built")
        return unraised, report
    return parts, report


def stands(parts: list[dict]) -> dict:
    from .kinematics import locked
    from .mechanism_bases import PIVOT_AS
    from .stability import stability

    st = stability(locked(parts, PIVOT_AS), vehicle=any(p["type"] == "wheel" for p in parts))
    return {"passed": st.passed, "issues": [i.message for i in st.issues][:4]}
