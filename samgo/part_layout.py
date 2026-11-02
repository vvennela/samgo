"""Spark reads where a hinged surface goes on a part of the model: which edge of the part is its trailing (hinge)
edge and where along that edge the surface sits - from the photo and a top view of the voxel model.

Spark (the knowing model) already names the mechanisms per region in words ("flap x2, trailing edge inboard").
This grounds them on the model: Spark sees the photo and a top view (x across, y up, numbered stud ticks; shaded by
height; the asked-about part tinted) and answers in the view's own coordinates:

    {"edge": "+x", "from": 6, "to": 10}

Code then turns that into a verified handoff (samgo.flap_sites: the site on that edge nearest Spark's stretch where
a plain answer passes the generic checks), which Glimmer builds and code checks. Nothing here knows the object.
Answers are cached per photo, model and question.
"""

import asyncio
import hashlib
import json
import re
from pathlib import Path

import numpy as np

PROMPT = """The first image is a photo of {obj}. The second is a top view of a LEGO-scale model of it: x runs across
(numbered ticks, one per stud), y runs up (numbered ticks), lighter is higher; the {part} is tinted orange.

We are adding a hinged {mech} to the {part} ({where}). From what you know about the real object, read on the top view
which edge of the orange part the {mech} hangs from, and where along that edge it sits. Answer with one JSON object,
nothing else:
{{"edge": "+x" | "-x" | "+y" | "-y", "from": <first tick along the edge>, "to": <last tick along the edge>}}
"+x" means the edge facing increasing x. "from"/"to" are on the axis that runs along that edge (y for an x edge,
x for a y edge)."""


def render_top(body: list[dict], tint: set[int], path: Path) -> None:
    """Top view (x across, y up) shaded by height, the parts `tint` in orange, a tick per stud."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .primitives import cells

    X = max(p["x"] + p["w"] for p in body)
    Y = max(p["y"] + p["l"] for p in body)
    top = np.full((X, Y), -1)
    who = np.full((X, Y), -1)
    for i, p in enumerate(body):
        for (x, y, z) in cells(p):
            if z >= top[x, y]:
                top[x, y], who[x, y] = z, i
    zmax = max(int(top.max()), 1)
    img = np.ones((Y, X, 3))
    for x in range(X):
        for y in range(Y):
            if top[x, y] < 0:
                continue
            g = 0.25 + 0.65 * top[x, y] / zmax
            img[y, x] = (min(1, g + 0.3), 0.45 + 0.3 * g, 0.1) if who[x, y] in tint else (g, g, g)
    fig, ax = plt.subplots(figsize=(max(6, X * 0.3), max(6, Y * 0.3)))
    ax.imshow(img, origin="lower", extent=(-0.5, X - 0.5, -0.5, Y - 0.5))
    ax.set_xticks(range(0, X, 2 if X > 30 else 1))
    ax.set_yticks(range(0, Y, 2 if Y > 30 else 1))
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_xlabel("x (studs)")
    ax.set_ylabel("y (studs)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
        out = {"edge": str(d["edge"]).strip(), "from": int(d["from"]), "to": int(d["to"])}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if out["edge"] not in ("+x", "-x", "+y", "-y"):
        return None
    out["from"], out["to"] = sorted((out["from"], out["to"]))
    return out


async def ask(photo: Path, body: list[dict], part_ids: list[int], work: Path, *, obj: str, part: str, mech: str,
              where: str, client=None) -> dict:
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    from .function_spec import BASE_URL, MODEL, _data_url

    work.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(Path(photo).read_bytes() + json.dumps([PROMPT, body, sorted(part_ids), obj, part, mech, where])
                         .encode()).hexdigest()[:16]
    cache = Path("runs/research_cache") / f"part_layout_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    view = work / "top_view.png"
    render_top(body, set(part_ids), view)
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    import os

    client = client or AsyncOpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=BASE_URL, max_retries=6)
    text = PROMPT.format(obj=f"a{'n' if obj[0] in 'aeiou' else ''} {obj}", part=part, mech=mech, where=where)
    out = {"error": "no answer"}
    for _ in range(3):  # an empty answer now and then (reasoning used the budget): ask again
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


def layout(photo: Path, body: list[dict], part_ids: list[int], work: Path, **kw) -> dict:
    return asyncio.run(ask(photo, body, part_ids, work, **kw))


# ---------------------------------------------------------------- Spark's answer -> a verified handoff

def to_frame(edge: str, X: int, Y: int) -> tuple[bool, bool, int]:
    """(swap, mirror, size) that turns `edge` into the +x edge, for samgo.primitives.frame/unframe."""
    swap = edge in ("+y", "-y")
    mirror = edge in ("-x", "-y")
    return swap, mirror, (Y if swap else X) + 1


def local_parts(body: list[dict], part_ids: list[int], margin: int = 3) -> list[int]:
    """The part's own parts and every part within `margin` studs of its box (coordinates unchanged), so site
    checks on a big model run on the neighbourhood: what a flap there touches, not the far side of the model."""
    x0 = min(body[i]["x"] for i in part_ids) - margin
    x1 = max(body[i]["x"] + body[i]["w"] for i in part_ids) + margin
    y0 = min(body[i]["y"] for i in part_ids) - margin
    y1 = max(body[i]["y"] + body[i]["l"] for i in part_ids) + margin
    near = {i for i, p in enumerate(body) if p["x"] < x1 and p["x"] + p.get("w", 1) > x0 and p["y"] < y1
            and p["y"] + p.get("l", 1) > y0}
    return sorted(near | set(part_ids))


def handoff(body: list[dict], part_ids: list[int], spark: dict, **sizes) -> dict | None:
    """The verified flap site on Spark's edge nearest Spark's stretch, in the model's own coordinates:
    {"edge", "axis", "rows": (lo, hi), "site" (in the turned frame), "frame"} or None if none is buildable."""
    from .flap_sites import pick, sites
    from .primitives import frame

    X = max(p["x"] + p["w"] for p in body)
    Y = max(p["y"] + p["l"] for p in body)
    swap, mirror, n = to_frame(spark["edge"], X, Y)
    turned = frame(body, swap, mirror, n)
    lo, hi = spark["from"], spark["to"]
    if len(body) > 600:  # a big model: search on the part and its neighbours only, and near Spark's stretch first
        keep = local_parts(turned, part_ids)
        sub = [turned[i] for i in keep]
        ids = [keep.index(i) for i in part_ids]
        found = sites(sub, ids, ys=(lo, hi), **sizes) or sites(sub, ids, **sizes)
    else:
        found = sites(turned, part_ids, **sizes)
    if not found:
        return None
    # the most overlap with Spark's stretch, then the biggest flap, then the nearest
    best = max(min(s["y1"], hi) - max(s["y0"], lo) for s in found)
    site = pick([s for s in found if min(s["y1"], hi) - max(s["y0"], lo) == best], near_y=(lo + hi) / 2)
    return {"edge": spark["edge"], "axis": "x" if not swap else "y", "rows": (site["y0"], site["y1"]),
            "site": site, "frame": [swap, mirror, n]}
