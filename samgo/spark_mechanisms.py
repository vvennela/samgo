"""Spark knows, Glimmer does, code verifies - for any object, not just the families with a hand-written builder.

1. Code splits the voxel model into regions (samgo.regions: a thick core and thin appendages; nothing object-specific)
   and draws them numbered.
2. Spark, from the photo and the numbered views, names each region and the hinged surfaces it has on the real object
   (its own knowledge: "a flap on the trailing edge, inboard"; "a lid along the back edge").
3. Spark grounds each surface on the model: which edge of the region it hangs from and where along it
   (samgo.part_layout).
4. Code turns that into a handoff it has checked is buildable (samgo.flap_sites: a site on that edge where a plain
   answer passes the generic checks).
5. Glimmer builds it from the local handoff (the parts near the site and a sentence: which edge, which rows), with
   the checker's precise feedback for up to `rounds` rounds.
6. Code checks the result on the whole model; a surface Glimmer cannot build is left out, never built by code.

Only what Glimmer has a generic skill for is attempted: hinged surfaces on an edge, swinging about a horizontal axis.
Everything else Spark names is reported as not supported yet.
"""

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

FLAP = [{"name": "flap", "kind": "swing", "axis": "horizontal", "degrees": 20, "count": 1}]
H = {"plate": 1, "brick": 3}

NAME_PROMPT = """The first image is a photo of {obj}. The second shows a LEGO-scale model of it split into numbered
regions (a top view and a side view; each region has one colour and its number at its centre).

For each numbered region, say which part of the real object it is, and list
- the hinged surfaces that part has on the real object: a panel along one edge of the part that swings about a hinge
  line along that edge;
- the parts on it that spin all the way round on an axle (not the wheels the object rolls on).
Use what you know about the real object; list only what really exists on it, and none if it has none. Answer with
one JSON object and nothing else:
{{"regions": {{"<number>": {{"part": "<name>",
                           "hinged": [{{"name": "<name>", "where": "<which edge, where along it, in words>",
                                       "axis": "horizontal" | "vertical"}}],
                           "spinning": [{{"name": "<name>", "where": "<which face of the part, in words>",
                                         "axis": "horizontal" | "vertical"}}]}}}}}}
"axis" is the hinge line's or spin axle's direction when the object stands as in the photo."""


def region_of_parts(parts: list[dict], labels: np.ndarray) -> list[int]:
    """The region label most of each part's cells fall in (0 if none)."""
    X, Y, Z = labels.shape
    out = []
    for p in parts:
        if p.get("type") not in H:
            out.append(0)
            continue
        cells = labels[max(p["x"], 0):min(p["x"] + p["w"], X), max(p["y"], 0):min(p["y"] + p["l"], Y),
                       max(p["z"], 0):min(p["z"] + H[p["type"]], Z)].ravel()
        cells = cells[cells > 0]
        out.append(int(np.bincount(cells).argmax()) if cells.size else 0)
    return out


async def name_regions(photo: Path, labels: np.ndarray, obj: str, work: Path, client=None) -> dict:
    """Spark's names and hinged surfaces per region (cached per photo, regions and prompt)."""
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    from .function_spec import BASE_URL, MODEL, _data_url
    from .regions import render

    work.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(Path(photo).read_bytes() + labels.tobytes() + str(labels.shape).encode()
                         + (NAME_PROMPT + obj).encode()).hexdigest()[:16]
    cache = Path("runs/research_cache") / f"regions_{key}.json"
    if cache.exists():  # JSON keys are strings; region numbers are ints
        return {"regions": {int(k): v for k, v in json.loads(cache.read_text())["regions"].items()}}
    view = work / "regions.png"
    render(labels, view)
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    import os

    client = client or AsyncOpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=BASE_URL, max_retries=6)
    text = NAME_PROMPT.format(obj=f"a{'n' if obj[:1] in 'aeiou' else ''} {obj}")
    out = {"error": "no answer"}
    for _ in range(3):
        r = await client.chat.completions.create(
            model=MODEL, max_tokens=8000, extra_body={"reasoning_effort": "low"},
            messages=[{"role": "user", "content": [{"type": "text", "text": text},
                                                   {"type": "image_url", "image_url": {"url": _data_url(photo)}},
                                                   {"type": "image_url",
                                                    "image_url": {"url": _data_url(view, 1400)}}]}])
        answer = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", answer, re.S)
        try:
            d = json.loads(m.group()) if m else None
            regions = {int(k): {"part": str(v.get("part", "")),
                                "hinged": [h for h in v.get("hinged", []) if isinstance(h, dict) and h.get("name")],
                                "spinning": [h for h in v.get("spinning", []) or []
                                             if isinstance(h, dict) and h.get("name")]}
                       for k, v in d["regions"].items()}
            out = {"regions": regions}
            break
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            out = {"error": f"unparseable answer: {answer[:200]}"}
    if "error" not in out:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return {"regions": {int(k): v for k, v in out["regions"].items()}} if "regions" in out else out


def attempt(body: list[dict], local: list[int], instruction: str, sampler, rounds: int = 3,
            max_tokens: int = 8000, timeout: int = 600, reqs: list[dict] = FLAP) -> dict:
    """Glimmer edits the local view; the edited whole model is checked; precise feedback between rounds."""
    from .edits import apply_edits
    from .free_edit_test import RULES, check, explain
    from .running_gear_task import FEEDBACK, edits_from

    chat = [{"role": "system", "content": RULES},
            {"role": "user", "content": instruction + "\nParts:\n" +
             "\n".join(f"{k}: {json.dumps(body[i])}" for k, i in enumerate(local))}]
    record = {"instruction": instruction, "parts_shown": len(local), "rounds": [], "passed": False}
    for rnd in range(1, rounds + 1):
        t, s, entry = time.time(), None, {"round": rnd}
        try:
            s = sampler(chat, max_tokens=max_tokens, temperature=0.0 if rnd == 1 else 0.7, timeout=timeout)
            entry.update(tokens=s.tokens, seconds=round(time.time() - t, 1), text=s.text)
            a = apply_edits(body, edits_from(s.text), local=local)
            v = check(body, a.parts, a.sources, reqs, stand=False)
            names = {"structure", "connected"} | {q["name"] for q in reqs}
            keep = {k: x for k, x in v.items() if k in names or k.endswith("_surfaces")}
            entry["passed"] = all(x[0] for x in keep.values())
            issues = explain(body, a.parts, a.sources, a.changed, keep, local)
            if entry["passed"]:
                edits = edits_from(s.text)
                record.update(passed=True, parts=a.parts, edits=[  # in the whole model's indices, to merge later
                    e | {"index": local[e["index"]]} if "index" in e else e for e in edits])
        except Exception as e:  # noqa: BLE001 - unparsable or invalid answers are verdicts too
            entry["passed"] = False
            issues = [f"answer: {type(e).__name__}: {e}"[:200]]
        entry["feedback"] = issues
        record["rounds"].append(entry)
        if entry["passed"]:
            break
        if s is not None:
            chat = chat + [{"role": "assistant", "content": s.text},
                           {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))}]
    return record


def build(photo: Path, target: np.ndarray, parts: list[dict], obj: str, sampler, work: Path, rounds: int = 3,
          max_surfaces: int = 2, gear: bool = True, glass: bool = True) -> tuple[list[dict], dict]:
    """Add the hinged surfaces Spark names that Glimmer can build, then the legs it stands on (samgo.spark_gear).
    Returns (parts, report)."""
    from .regions import split

    labels = split(target)
    scale = max(1.0, max(target.shape[:2]) / 40)  # surfaces sized to the model: a 92-stud model gets longer flaps
    loop = asyncio.new_event_loop()  # one loop for every Spark call of this build (its HTTP client lives in it)
    try:
        parts, report = _build(photo, parts, obj, sampler, work, rounds, max_surfaces, labels, loop, scale)
        if gear:
            from . import spark_gear

            parts, report["gear"] = spark_gear.build(photo, parts, obj, sampler, work / "gear", loop, rounds=rounds)
        if glass:  # last: only colours and cuts, after every mechanism is in
            from . import spark_glass

            parts, report["glass"] = spark_glass.build(photo, parts, obj, work / "glass", loop)
        (work / "spark_mechanisms.json").write_text(json.dumps(report, indent=1, default=str))
        return parts, report
    finally:
        loop.close()


def _build(photo, parts, obj, sampler, work, rounds, max_surfaces, labels, loop, scale=1.0, workers: int = 8):
    """Every mechanism Spark names, in parallel against the same model: Spark's layouts all at once, the spots,
    then Glimmer's attempts (each with its own feedback rounds) at once; the passing edit lists are merged and the
    merged model is checked again as a whole (one at a time, in Spark's order, if they conflict)."""
    from concurrent.futures import ThreadPoolExecutor

    from .flap_data import _view, instruction
    from .part_layout import ask as ask_layout
    from .part_layout import handoff
    from .primitives import frame
    from .rotor_data import ROTOR, rotor_site
    from .rotor_data import instruction as rotor_instruction

    named = loop.run_until_complete(name_regions(photo, labels, obj, work))
    report = {"regions": named.get("regions"), "error": named.get("error"), "surfaces": [], "unsupported": [],
              "rotors": []}
    if "regions" not in named:
        return parts, report
    region_of = region_of_parts(parts, labels)
    jobs = []  # (kind, entry, ids, layout coroutine args)
    for r, d in sorted(named["regions"].items()):
        for h in d["hinged"]:
            if h.get("axis") != "horizontal":
                report["unsupported"].append({"region": r, "part": d["part"], **h, "why": "only horizontal hinges"})
                continue
            entry = {"region": r, "part": d["part"], **h}
            report["surfaces"].append(entry)
            jobs.append(("surface", entry, [i for i, x in enumerate(region_of) if x == r],
                         dict(mech=h["name"], folder=f"layout_{r}_{h['name']}")))
        for h in d.get("spinning", []):
            entry = {"region": r, "part": d["part"], **h}
            if len(report["rotors"]) >= 4:
                break
            report["rotors"].append(entry)
            if h.get("axis") == "vertical":
                entry["error"] = "only horizontal axles yet"
                continue
            jobs.append(("rotor", entry, [i for i, x in enumerate(region_of) if x == r],
                         dict(mech=f"{h['name']} (a part that spins on an axle)", folder=f"rotor_{r}_{h['name']}")))
    jobs = [j for j in jobs if j[2] or j[1].update(error="no parts in the region")]

    async def layouts():
        return await asyncio.gather(*[ask_layout(photo, parts, ids, work / a["folder"], obj=obj, part=e["part"],
                                                 mech=a["mech"], where=e.get("where", "")) for _, e, ids, a in jobs])

    sizes = dict(lengths=tuple(sorted({round(n * scale) for n in (3, 4, 5)})),
                 chords=tuple(sorted({round(n * scale) for n in (2, 3, 4)})))
    tasks = []  # (entry, view, text, reqs)
    for (kind, entry, ids, _), sp in zip(jobs, loop.run_until_complete(layouts())):
        entry["spark_layout"] = sp
        if kind == "surface":
            hand = handoff(parts, ids, sp, **sizes) if "error" not in sp else None
            if hand is None:
                entry["error"] = "no buildable site on that edge"
                continue
            swap, mirror, n = hand["frame"]
            entry["handoff"] = {"edge": hand["edge"], "rows": hand["rows"]}
            tasks.append((entry, _view(frame(parts, swap, mirror, n), hand["site"], "L2", set(ids)),
                          instruction("L2", obj, hand["site"], (swap, mirror, n)), FLAP))
        else:
            site = rotor_site(parts, ids, sp["edge"], sp["from"], sp["to"]) if "error" not in sp else None
            if site is None:
                entry["error"] = "no buildable spot on that face"
                continue
            entry["site"] = site
            tasks.append((entry, _rotor_view(parts, site), rotor_instruction("L2", obj, site), ROTOR))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        recs = list(pool.map(lambda t: attempt(parts, t[1], t[2], sampler, rounds=rounds, reqs=t[3]), tasks))
    for (entry, _, text, _), rec in zip(tasks, recs):
        entry.update(passed=rec["passed"], rounds=rec["rounds"], instruction=text)
    passing = [(e, rec, q) for (e, _, _, q), rec in zip(tasks, recs) if rec["passed"]]
    surfaces = [p for p in passing if p[2] is FLAP][:max_surfaces]
    passing = surfaces + [p for p in passing if p[2] is not FLAP]
    parts = _merge(parts, passing, report)
    (work / "spark_mechanisms.json").write_text(json.dumps(report, indent=1, default=str))
    return parts, report


def _merge(body, passing, report):
    """All passing edit lists applied to the model at once and checked as a whole; if they conflict, added one at a
    time in order, each kept only if the whole model still passes everything kept so far."""
    from .edits import apply_edits
    from .free_edit_test import check

    def verdict(chosen):
        try:
            a = apply_edits(body, [e for _, rec, _ in chosen for e in rec["edits"]])
        except Exception:  # noqa: BLE001 - two edit lists touch the same part
            return None
        v = check(body, a.parts, a.sources, _merged_reqs(chosen), stand=False)
        keep = {k: x for k, x in v.items() if k != "stands"}
        return a.parts if all(x[0] for x in keep.values()) else None

    if not passing:
        return body
    together = verdict(passing)
    if together is not None:
        return together
    kept, parts = [], body
    for item in passing:
        out = verdict(kept + [item])
        if out is None:
            item[0].update(passed=False, merge_error="conflicts with a mechanism built before it")
            continue
        kept.append(item)
        parts = out
    return parts


def _merged_reqs(chosen):
    """One requirement list for several mechanisms: counts add up per kind and axis."""
    out = {}
    for _, _, q in chosen:
        for r in q:
            key = (r["kind"], r.get("axis"), r.get("name") if r["kind"] not in ("rotor", "swing") else None)
            if key in out:
                out[key] = out[key] | {"count": out[key].get("count", 1) + r.get("count", 1)}
            else:
                out[key] = dict(r)
    return list(out.values())


def _rotor_view(parts, site):
    from .rotor_data import _view

    return _view(parts, site, "L2")


def main() -> None:
    """Run on a folder with photo.png, body40.json, target40.npz (the sweep's inputs) - e.g. a held-out test."""
    import argparse

    from dotenv import load_dotenv

    from .assembly import FireworksSampler

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--object", default=None, help="what it is (default: the folder's research.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=40, help="which body<size>.json / target<size>.npz")
    ap.add_argument("--no-gear", action="store_true")
    ap.add_argument("--no-glass", action="store_true")
    args = ap.parse_args()
    body = json.loads((args.folder / f"body{args.size}.json").read_text())
    target = np.load(args.folder / f"target{args.size}.npz")
    target = target[target.files[0]]
    obj = args.object or json.loads((args.folder / "research.json").read_text()).get("object", "object")
    args.out.mkdir(parents=True, exist_ok=True)
    parts, report = build(args.folder / "photo.png", target, body, obj, FireworksSampler(args.model), args.out,
                          gear=not args.no_gear, glass=not args.no_glass)
    (args.out / "build.json").write_text(json.dumps(parts))
    for s in report["surfaces"]:
        print(f"region {s['region']} {s['part']!r}: {s['name']} ({s.get('where')}) -> "
              f"{s.get('handoff')} passed={s.get('passed')} {s.get('error', '')}")
    print("unsupported:", [(u["part"], u["name"], u.get("axis")) for u in report["unsupported"]])
    for r in report.get("rotors", []):
        print(f"rotor region {r['region']} {r['part']!r}: {r['name']} ({r.get('where')}) -> {r.get('site')} "
              f"passed={r.get('passed')} {r.get('error', '')}")
    for g in report.get("gear", {}).get("legs", []):
        print(f"leg {g['name']!r} at ({g['x']}, {g['y']}) -> {g.get('site')} passed={g.get('passed')} "
              f"{g.get('error', '')}")
    if "gear" in report:
        print("stands on its wheels:", report["gear"].get("stands"))
    if "glass" in report:
        print("glass:", {k: v for k, v in report["glass"].items() if k != "spark"})


if __name__ == "__main__":
    main()
