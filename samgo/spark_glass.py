"""Glass wherever the real object is transparent - Spark knows where, code glazes, for any object.

Spark sees the photo and two views of the LEGO-scale model (top: x across, y up; side: x across, z up; numbered
ticks) and lists the transparent parts of the real object (a canopy, windows, a windshield, a lamp cover, a display
case) as boxes in the views' coordinates. Code makes the model's visible cells inside those boxes glass
(samgo.glass.glaze: whole parts recolored, plates across a glass edge cut, only sizes LEGO made in trans-clear, the
build held together as before). Only colors and cuts change; every structural and motion check is unaffected.
Nothing here knows what the object is.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

PROMPT = """The first image is a photo of {obj}. The second shows a LEGO-scale model of it from the top (x across,
y up) and from the side (x across, z up), with numbered ticks.

From what you know about the real object: which of its parts are transparent glass or clear plastic (windows, a
windshield, a canopy, a cover)? For each, give the box it fills on the model, read on the views: x and y on the top
view, z on the side view. Answer with one JSON object, nothing else:
{{"glass": [{{"name": "<name>", "x": [<from>, <to>], "y": [<from>, <to>], "z": [<from>, <to>]}}]}}
An empty list if the real object has no transparent parts."""


def factor(parts: list[dict], cap: int = 60) -> int:
    """Big models are shown to Spark reduced to about `cap` cells a side (its boxes are scaled back)."""
    from .glass import BODY_TYPES, _cells

    cells = [c for p in parts if p.get("type") in BODY_TYPES for c in _cells(p)]
    span = max(max(c[k] for c in cells) + 1 for k in range(3))
    return max(1, -(-span // cap))


def render_views(parts: list[dict], path: Path, f: int = 1) -> None:
    """Top view (x, y) and side view (x, z), shaded by depth, a tick every stud / plate (every `f` of them, as one
    cell, on a reduced view)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .glass import BODY_TYPES, _cells

    cells = {(x // f, y // f, z // f) for p in parts if p.get("type") in BODY_TYPES for (x, y, z) in _cells(p)}
    X = max(x for x, _, _ in cells) + 1
    Y = max(y for _, y, _ in cells) + 1
    Z = max(z for _, _, z in cells) + 1
    top = np.full((Y, X), -1.0)
    side = np.full((Z, X), -1.0)
    for x, y, z in cells:
        top[y, x] = max(top[y, x], z)
        side[z, x] = max(side[z, x], Y - y)  # nearer the viewer (low y) is lighter
    fig, axes = plt.subplots(1, 2, figsize=(max(10, X * 0.5), max(5, max(Y, Z) * 0.3)))
    for ax, img, depth, ylab, n in ((axes[0], top, Z, "y (studs)", Y), (axes[1], side, Y, "z (plates)", Z)):
        shade = np.where(img < 0, 1.0, 0.25 + 0.65 * img / max(depth, 1))
        ax.imshow(shade, origin="lower", cmap="gray", vmin=0, vmax=1, extent=(-0.5, X - 0.5, -0.5, n - 0.5))
        ax.set_xticks(range(0, X, 2 if X > 30 else 1))
        ax.set_yticks(range(0, n, 2 if n > 30 else 1))
        ax.tick_params(labelsize=6)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.set_xlabel("x (studs)")
        ax.set_ylabel(ylab)
    axes[0].set_title("top view")
    axes[1].set_title("side view")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# Transparent parts that face up (a canopy, a roof window): glazed on top too; every other window only from the side.
TOP_GLASS = re.compile(r"canopy|roof|sun ?roof|sky ?light|dome|bubble|top", re.I)


NAMES_PROMPT = """This is a photo of {obj}. From what you know about the real object: which of its parts are
transparent glass or clear plastic and visible in this photo? Name each as a short noun phrase an image segmenter can
find (like "windshield", "side windows", "cockpit windows", "headlight covers"). Answer with one JSON object, nothing
else: {{"glass": ["<phrase>", ...]}} - an empty list if none."""


def names(photo: Path, obj: str) -> list[str]:
    """Spark's names for the object's transparent parts (cached per photo and object); [] if none or no answer."""
    import asyncio
    import os

    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    from .function_spec import BASE_URL, MODEL, _data_url

    key = hashlib.sha256(Path(photo).read_bytes() + (NAMES_PROMPT + obj).encode()).hexdigest()[:16]
    cache = Path("runs/research_cache") / f"glass_names_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    async def ask():
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        client = AsyncOpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=BASE_URL, max_retries=6)
        text = NAMES_PROMPT.format(obj=f"a{'n' if obj[:1] in 'aeiou' else ''} {obj}")
        for _ in range(3):
            r = await client.chat.completions.create(
                model=MODEL, max_tokens=2000, extra_body={"reasoning_effort": "low"},
                messages=[{"role": "user", "content": [{"type": "text", "text": text},
                                                       {"type": "image_url", "image_url": {"url": _data_url(photo)}}]}])
            m = re.search(r"\{.*\}", r.choices[0].message.content or "", re.S)
            try:
                return [str(g)[:40] for g in json.loads(m.group())["glass"]][:6]
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        return None

    out = asyncio.run(ask())
    if out is None:
        return []
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def glaze_cells(parts: list[dict], cells: set, cut: bool = True) -> tuple[list[dict], dict]:
    """Glass exactly where the photo shows it: the visible cells SAM 3.1 marked (samgo.voxelize marker)."""
    from .glass import glaze, visible_cells

    win = visible_cells(parts) & set(cells)
    out, n = glaze(parts, win, cut=cut)
    return out, {"source": "sam", "cells": len(win), "glass_parts": n}


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group()) if m else None
        out = []
        for g in d["glass"]:
            box = {k: sorted(int(v) for v in g[k][:2]) for k in ("x", "y", "z")}
            out.append({"name": str(g.get("name", "glass"))} | box)
        return {"glass": out}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError):
        return None


async def ask(photo: Path, parts: list[dict], work: Path, *, obj: str, client=None) -> dict:
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    from .function_spec import BASE_URL, MODEL, _data_url

    work.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(Path(photo).read_bytes() + json.dumps([PROMPT, parts, obj]).encode()).hexdigest()[:16]
    cache = Path("runs/research_cache") / f"glass_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    view = work / "views.png"
    f = factor(parts)
    render_views(parts, view, f)
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
                                                    "image_url": {"url": _data_url(view, 1600)}}]}])
        answer = r.choices[0].message.content or ""
        out = parse(answer) or {"error": f"unparseable answer: {answer[:200]}"}
        if "error" not in out:
            if f > 1:  # back to the model's own cells
                out = {"glass": [g | {k: [g[k][0] * f, (g[k][1] + 1) * f - 1] for k in ("x", "y", "z")}
                                 for g in out["glass"]], "view_factor": f}
            break
    if "error" not in out:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out


def apply(parts: list[dict], spark: dict, cut: bool = True) -> tuple[list[dict], dict]:
    """Glaze the visible cells inside Spark's boxes. Returns (parts, report)."""
    from .glass import glaze, visible_cells

    if "glass" not in spark:
        return parts, {"error": spark.get("error")}
    from .glass import BODY_TYPES, _cells

    seen = visible_cells(parts)
    filled = {c for p in parts if p.get("type") in BODY_TYPES for c in _cells(p)}

    def sideways(c):  # seen from a side (an open neighbour in x or y): windows, windshields - not a roof
        x, y, z = c
        return any((x + dx, y + dy, z) not in filled for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    win = {c for c in seen for g in spark["glass"]
           if g["x"][0] <= c[0] <= g["x"][1] and g["y"][0] <= c[1] <= g["y"][1] and g["z"][0] <= c[2] <= g["z"][1]
           and (TOP_GLASS.search(g.get("name", "")) or sideways(c))}
    out, n = glaze(parts, win, cut=cut)
    return out, {"boxes": spark["glass"], "cells": len(win), "glass_parts": n}


def build(photo: Path, parts: list[dict], obj: str, work: Path, loop, cut: bool = True) -> tuple[list[dict], dict]:
    """`cut=False`: whole parts only (a model whose hinged parts are already in place keeps every part as is)."""
    sp = loop.run_until_complete(ask(photo, parts, work, obj=obj))
    out, report = apply(parts, sp, cut=cut)
    return out, report | {"spark": sp}
