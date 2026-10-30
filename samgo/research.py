"""Research step: photo -> what is it -> how do its parts move (Wikipedia) -> requirements the verifiers can check.

    uv run python -m samgo.research photo.jpg
    uv run python -m samgo.research --name "office chair"

1. Identify: Muse Spark Open names the object in the photo (function_spec's call; skipped with --name).
2. Look up: Wikipedia search + extracts of the best articles (cached in runs/research_cache/). If the fetch
   fails the model answers from its own knowledge and the source is marked "model".
3. Requirements: the model turns name + text into moving parts. The answer is validated here (only the motion
   kinds kinematics.requirement_met() understands: "spin" and "swing" with a vertical/horizontal axis) and
   retried once with the validation errors. Anything else goes in "unsupported" with a reason.

The output's "requirements" list is ready for kinematics.requirement_met():
    {"kind": "spin", "count": 4}  /  {"kind": "swing", "axis": "vertical", "degrees": 90}
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .function_spec import BASE_URL, MODEL, _data_url
from .function_spec import infer as identify_photo

CACHE = Path(__file__).resolve().parent.parent / "runs" / "research_cache"
USER_AGENT = "SAMgo/0.1 (hackathon research tool)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
MOTIONS = ("spin", "swing")
AXES = ("vertical", "horizontal")

PROMPT = """You plan working mechanisms for a LEGO model of a real object. Object: "{name}".

Wikipedia articles found for it (some may be about a different sense of the word; use only the ones about
this object, and fill in from your own knowledge what they leave out):
{text}

List the object's real moving parts that a small LEGO model could reproduce. Only two motions can be checked:
- "spin": a wheel/roller turning freely on an axle, many turns (range_deg 360). Use for road wheels, caster
  wheels, bicycle wheels, cranks, fans.
- "swing": a part turning on a pivot through a limited or full angle. axis "vertical" (pivot pin points up:
  swivel seats, turntables, side-hinged doors, steering) or "horizontal" (pivot pin lies flat: laptop lids,
  top-hinged doors, levers, flaps).
A part that moves in two ways is two entries: a caster wheel spins on its axle AND its fork swivels on a
vertical pin; a bicycle front wheel spins AND the handlebar/fork steers. Give range_deg as how far it really turns in one direction (360 for a full swivel). count = how many identical
parts move this way (5 caster wheels -> count 5). lego_mechanism = the LEGO way to build it, e.g.
"turntable 3680/3679", "axle + wheel", "hinge brick 3937/3938", "click hinge", "technic pin".
Motions that are not a rotation on one pin (sliding drawers, telescoping, springs, gas lifts, buttons,
flexing) go in "unsupported" with a short reason.

Answer with one JSON object and nothing else:
{{"object": "<short name>",
 "moving_parts": [{{"name": "...", "motion": "spin"|"swing", "axis": "vertical"|"horizontal",
                   "range_deg": <number>, "count": <integer>, "lego_mechanism": "..."}}],
 "unsupported": [{{"name": "...", "reason": "..."}}],
 "articles_used": ["<titles of the articles above that are about this object>"]}}"""


# ---------------------------------------------------------------- look up

def _get(params: dict) -> dict:
    url = WIKI_API + "?" + urllib.parse.urlencode(params | {"format": "json", "formatversion": 2})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for wait in (2, 5, 10, None):  # Wikipedia rate-limits bursts with 429
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or wait is None:
                raise
            time.sleep(wait)


def lookup(name: str, articles: int = 3) -> dict:
    """{"text", "sources", "source"}: the lead sections of the top Wikipedia hits for `name`, cached on disk.
    Several hits because the first can be the wrong sense ("microwave" -> the radiation, not the oven); the
    model says which ones it used."""
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60] or hashlib.md5(name.encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        hits = _get({"action": "query", "list": "search", "srsearch": name, "srlimit": articles + 2})
        titles = [h["title"] for h in hits["query"]["search"] if "disambiguation" not in h["title"]][:articles]
        if not titles:
            raise LookupError("no articles")
        found = _get({"action": "query", "prop": "extracts|info", "inprop": "url", "explaintext": 1, "exintro": 1,
                      "exlimit": articles, "titles": "|".join(titles), "redirects": 1})["query"]["pages"]
        order = {t: i for i, t in enumerate(titles)}
        pages = sorted((p for p in found if p.get("extract")), key=lambda p: order.get(p["title"], 99))
        if not pages:
            raise LookupError("no extracts")
        result = {"text": "\n\n".join(f"## {p['title']}\n{p['extract'][:2500]}" for p in pages),
                  "sources": {p["title"]: p["fullurl"] for p in pages}, "source": "wikipedia"}
    except Exception as e:  # network down, API change: fall back to the model's knowledge, don't cache
        return {"text": "(no web text; use your own knowledge)", "sources": {}, "source": "model",
                "error": f"{type(e).__name__}: {e}"}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    return result


# ---------------------------------------------------------------- validate

def parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        out = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def validate(spec) -> list[str]:
    """Problems with a requirements answer; [] means it is usable."""
    if not isinstance(spec, dict):
        return ["answer is not a JSON object"]
    errors = []
    if not isinstance(spec.get("object"), str) or not spec["object"].strip():
        errors.append('"object" must be a non-empty string')
    parts = spec.get("moving_parts")
    if not isinstance(parts, list):
        return errors + ['"moving_parts" must be a list']
    for i, p in enumerate(parts):
        where = f"moving_parts[{i}]"
        if not isinstance(p, dict):
            errors.append(f"{where} must be an object")
            continue
        if not isinstance(p.get("name"), str) or not p["name"].strip():
            errors.append(f"{where}.name must be a non-empty string")
        if p.get("motion") not in MOTIONS:
            errors.append(f"{where}.motion is {p.get('motion')!r}; allowed: {', '.join(MOTIONS)} "
                          "(move anything else to unsupported)")
        if p.get("axis") not in AXES:
            errors.append(f"{where}.axis is {p.get('axis')!r}; allowed: {', '.join(AXES)}")
        deg = p.get("range_deg")
        if isinstance(deg, bool) or not isinstance(deg, (int, float)) or not 0 < deg <= 360:
            errors.append(f"{where}.range_deg must be a number in (0, 360]")
        count = p.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            errors.append(f"{where}.count must be an integer >= 1")
        if not isinstance(p.get("lego_mechanism"), str) or not p["lego_mechanism"].strip():
            errors.append(f"{where}.lego_mechanism must be a non-empty string")
    unsupported = spec.get("unsupported", [])
    if not isinstance(unsupported, list):
        errors.append('"unsupported" must be a list')
    else:
        for i, u in enumerate(unsupported):
            if not (isinstance(u, dict) and isinstance(u.get("name"), str) and isinstance(u.get("reason"), str)):
                errors.append(f'unsupported[{i}] must be {{"name": str, "reason": str}}')
    return errors


def requirements(parts: list[dict]) -> list[dict]:
    """The moving parts as kinematics.requirement_met() requirement dicts."""
    reqs = []
    for p in parts:
        if p["motion"] == "spin":
            reqs.append({"kind": "spin", "count": p["count"], "part": p["name"]})
        else:
            reqs.append({"kind": "swing", "axis": p["axis"], "degrees": p["range_deg"], "count": p["count"],
                         "part": p["name"]})
    return reqs


# ---------------------------------------------------------------- model

def _client():
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv()
    return AsyncOpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=BASE_URL, max_retries=6)


async def plan(name: str, found: dict, client, photo: Path | None = None) -> tuple[dict, int]:
    """(validated spec, model calls made). Retries once with the validation errors."""
    content = [{"type": "text", "text": PROMPT.format(name=name, text=found["text"])}]
    if photo is not None:
        content.append({"type": "image_url", "image_url": {"url": _data_url(photo)}})
    messages = [{"role": "user", "content": content}]
    calls, errors, spec = 0, [], None
    for _ in range(2):
        r = await client.chat.completions.create(model=MODEL, max_tokens=4000, messages=messages,
                                                 extra_body={"reasoning_effort": "low"})
        calls += 1
        text = r.choices[0].message.content or ""
        spec = parse_json(text)
        errors = validate(spec) if spec is not None else ["no JSON object found in the answer"]
        if not errors:
            return spec, calls
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "That answer is invalid:\n- " + "\n- ".join(errors)
                      + "\nReply with the corrected JSON object only."}]
    return {"object": name, "moving_parts": [], "unsupported": [], "error": "; ".join(errors),
            "raw": spec}, calls


async def research(photo: Path | None = None, name: str | None = None, client=None) -> dict:
    client = client or _client()
    calls = 0
    if name is None:
        if photo is None:
            raise ValueError("need a photo or a name")
        name = (await identify_photo(photo, client)).get("object") or "object"
        calls += 1
    found = lookup(name)
    spec, n = await plan(name, found, client, photo)
    calls += n
    spec.setdefault("unsupported", [])
    spec["requirements"] = requirements(spec["moving_parts"]) if "error" not in spec else []
    used = spec.pop("articles_used", None)
    used = used if isinstance(used, list) else list(found["sources"])
    spec["sources"] = [url for title, url in found["sources"].items() if title in used]
    spec["source"] = found["source"]
    spec["model_calls"] = calls
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo", nargs="?", type=Path)
    ap.add_argument("--name", help="object name; skips identifying it from the photo")
    args = ap.parse_args()
    if args.photo is None and args.name is None:
        ap.error("give a photo or --name")
    print(json.dumps(asyncio.run(research(args.photo, args.name)), indent=1))


if __name__ == "__main__":
    main()
