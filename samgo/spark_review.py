"""Spark reviews the build: the brain checks the finished model against the real object, not only the photo.

Spark labels the object before anything is built (samgo.spark_mechanisms); nothing used to check the result
against it, so a model could pass every physical check and still miss two of four engines, put a fan outside its
nacelle, or have a red windshield. Here Spark sees the photo, the region map and the model as built (top, side and
front views in its true colors) with the list of working parts built so far, and answers with a punch list in a
fixed vocabulary. Code carries each item out and the checker verifies the result like any answer:

- phase 1, working parts: "add" a missing hinged or spinning part (built by Glimmer through the usual Spark
  layout -> spot -> Glimmer -> checker route) and "remove" a built one that is not on the real object or clearly in
  the wrong place (its verified edit list is dropped and the rest merged again);
- phase 2, appearance: "paint" a box of the model in a LEGO color, and "glass" a box that is transparent on the real
  object. Paint never changes a part's shape; glass cuts plates only where nothing hinges.

Nothing here knows what the object is: the vocabulary is the same for a fan, a car or an airliner.
"""

import json
import re
from pathlib import Path

import numpy as np

MECH_PROMPT = """The first image is a photo of {obj}. The second shows a LEGO model of it split into numbered regions
(top and side views). The third shows the LEGO model as built now (top, side and front views, its real colors; ticks
count cells{scale}).

Working parts built so far (boxes in the third image's coordinates):
{built}

From what you know about the real object, compare it with the model: which working parts does the real object have
that the model is missing (a hinged panel that swings on an edge, or a part that spins all the way round on an axle,
like a propeller, fan or rotor), and which built ones are wrong (the real object has no such part there, or it is
clearly in the wrong place)? Use the region numbers of the second image. Answer with one JSON object, nothing else:
{{"add": [{{"region": <number>, "kind": "hinged" | "spinning", "name": "<name>", "where": "<where on that part, in
words>", "axis": "horizontal" | "vertical"}}],
 "remove": [{{"id": <number>, "why": "<why>"}}]}}
Empty lists if the working parts match the real object."""

LOOK_PROMPT = """The first image is a photo of {obj}. The second shows the LEGO model of it as built now (top, side and
front views, its real colors; ticks count cells{scale}). Glass is light blue.

From what you know about the real object, compare it with the model:
- which areas have the wrong color? Give each as a box and one of these colors: {colors}.
- which transparent parts of the real object (windshield, windows, canopy, a clear cover) are not glass yet? Give each
  as a box.
Boxes in the image's coordinates: x and y read on the top view, z on the side view. Answer with one JSON object,
nothing else:
{{"paint": [{{"name": "<what>", "color": "<color>", "x": [<from>, <to>], "y": [<from>, <to>], "z": [<from>, <to>]}}],
 "glass": [{{"name": "<what>", "x": [<from>, <to>], "y": [<from>, <to>], "z": [<from>, <to>]}}]}}
Empty lists if the model already matches."""

GLASS_RGB = (190, 225, 245)
OTHER_RGB = (240, 150, 40)  # Technic and joint parts: visible as working parts


def _cells(p: dict):
    h = {"plate": 1, "brick": 3, "technic_brick": 3}.get(p.get("type"), 1)
    for x in range(p["x"], p["x"] + p["w"]):
        for y in range(p["y"], p["y"] + p["l"]):
            for z in range(p["z"], p["z"] + h):
                yield x, y, z


def _rgb(p: dict):
    from .parts import COLORS

    c = p.get("color")
    if c in ("trans_clear", "trans_black"):
        return GLASS_RGB
    if p.get("type") not in ("plate", "brick"):
        return OTHER_RGB
    return COLORS.get(c, (None, (150, 150, 150)))[1]


def factor(parts: list[dict], cap: int = 60) -> int:
    cells = [c for p in parts if "w" in p and "l" in p for c in _cells(p)]
    span = max(max(c[k] for c in cells) - min(c[k] for c in cells) + 1 for k in range(3))
    return max(1, -(-span // cap))


def render(parts: list[dict], path: Path, f: int = 1) -> None:
    """Top (x, y), side (x, z, seen from -y) and front (y, z, seen from -x) views in the parts' colors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {}
    for p in parts:
        if "w" in p and "l" in p:
            for x, y, z in _cells(p):
                color[(x // f, y // f, z // f)] = _rgb(p)
    X = max(c[0] for c in color) + 1
    Y = max(c[1] for c in color) + 1
    Z = max(c[2] for c in color) + 1
    top, side, front = (np.ones((Y, X, 3)), np.ones((Z, X, 3)), np.ones((Z, Y, 3)))
    best_top, best_side, best_front = {}, {}, {}
    for (x, y, z), rgb in color.items():
        if z >= best_top.get((x, y), -1):
            best_top[(x, y)] = z
            top[y, x] = np.array(rgb) / 255
        if y <= best_side.get((x, z), 1e9):
            best_side[(x, z)] = y
            side[z, x] = np.array(rgb) / 255
        if x <= best_front.get((y, z), 1e9):
            best_front[(y, z)] = x
            front[z, y] = np.array(rgb) / 255
    fig, axes = plt.subplots(1, 3, figsize=(max(12, (2 * X + Y) * 0.25), max(5, max(Y, Z) * 0.3)))
    for ax, img, w, h, title, xl, yl in ((axes[0], top, X, Y, "top view", "x", "y"),
                                         (axes[1], side, X, Z, "side view", "x", "z"),
                                         (axes[2], front, Y, Z, "front view", "y", "z")):
        ax.imshow(img, origin="lower", extent=(-0.5, w - 0.5, -0.5, h - 0.5))
        ax.set_xticks(range(0, w, 2 if w > 30 else 1))
        ax.set_yticks(range(0, h, 2 if h > 30 else 1))
        ax.tick_params(labelsize=6)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mech_box(edits: list[dict]) -> dict | None:
    """The box of the parts an edit list adds (cells, model coordinates)."""
    pts = []
    for e in edits:
        p = e.get("part") if e.get("op") in ("add", "set") else None
        if p and all(k in p for k in ("x", "y", "z")):
            pts.append((p["x"], p["y"], p["z"]))
            if "w" in p and "l" in p:
                pts.append((p["x"] + p["w"] - 1, p["y"] + p["l"] - 1, p["z"]))
    if not pts:
        return None
    a = np.array(pts)
    return {k: [int(a[:, n].min()), int(a[:, n].max())] for n, k in enumerate("xyz")}


def _json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        return None


def parse_mech(text: str, regions: dict, ids: set) -> dict | None:
    d = _json(text)
    if not isinstance(d, dict):
        return None
    add, remove = [], []
    for a in d.get("add") or []:
        try:
            r = int(a["region"])
            if r in regions and a.get("kind") in ("hinged", "spinning") and a.get("name"):
                add.append({"region": r, "kind": a["kind"], "name": str(a["name"]), "where": str(a.get("where", "")),
                            "axis": "vertical" if a.get("axis") == "vertical" else "horizontal"})
        except (KeyError, TypeError, ValueError):
            continue
    for r in d.get("remove") or []:
        try:
            if int(r["id"]) in ids:
                remove.append({"id": int(r["id"]), "why": str(r.get("why", ""))})
        except (KeyError, TypeError, ValueError):
            continue
    return {"add": add, "remove": remove}


def parse_look(text: str, f: int) -> dict | None:
    from .parts import COLORS

    d = _json(text)
    if not isinstance(d, dict):
        return None

    def box(g):
        return {k: [min(int(v) for v in g[k][:2]) * f, (max(int(v) for v in g[k][:2]) + 1) * f - 1] for k in "xyz"}

    out = {"paint": [], "glass": []}
    for g in d.get("paint") or []:
        try:
            if g.get("color") in COLORS:
                out["paint"].append({"name": str(g.get("name", "")), "color": g["color"]} | box(g))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    for g in d.get("glass") or []:
        try:
            out["glass"].append({"name": str(g.get("name", ""))} | box(g))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return out


async def _ask(images: list[Path], text: str, parse, client=None) -> dict:
    import os

    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    from .function_spec import BASE_URL, MODEL, _data_url

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    client = client or AsyncOpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=BASE_URL, max_retries=6)
    content = [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": _data_url(p, 1600) if n else _data_url(p)}}
        for n, p in enumerate(images)]
    answer = ""
    for _ in range(3):
        r = await client.chat.completions.create(model=MODEL, max_tokens=8000, extra_body={"reasoning_effort": "low"},
                                                 messages=[{"role": "user", "content": content}])
        answer = r.choices[0].message.content or ""
        out = parse(answer)
        if out is not None:
            return out
    return {"error": f"unparseable answer: {answer[:200]}"}


def _article(obj: str) -> str:
    return f"a{'n' if obj[:1].lower() in 'aeiou' else ''} {obj}"


def ask_mechanisms(photo: Path, regions_png: Path, parts: list[dict], obj: str, built: list[dict], regions: dict,
                   work: Path, loop) -> dict:
    """Spark's punch list for the working parts. `built`: [{"id", "name", "kind", "region", "box"}]."""
    work.mkdir(parents=True, exist_ok=True)
    f = factor(parts)
    view = work / "model_views.png"
    render(parts, view, f)
    lines = []
    for b in built:
        bx = b.get("box")
        where = (f"x {bx['x'][0] // f}-{bx['x'][1] // f}, y {bx['y'][0] // f}-{bx['y'][1] // f}, "
                 f"z {bx['z'][0] // f}-{bx['z'][1] // f}") if bx else "position unknown"
        lines.append(f"#{b['id']} region {b['region']}: {b['name']} ({b['kind']}) at {where}")
    text = MECH_PROMPT.format(obj=_article(obj), scale=f", one tick = {f} cells" if f > 1 else "",
                              built="\n".join(lines) or "(none)")
    return loop.run_until_complete(_ask([photo, regions_png, view], text,
                                        lambda t: parse_mech(t, regions, {b["id"] for b in built})))


def ask_look(photo: Path, parts: list[dict], obj: str, work: Path, loop) -> dict:
    from .parts import COLORS

    work.mkdir(parents=True, exist_ok=True)
    f = factor(parts)
    view = work / "model_views.png"
    render(parts, view, f)
    text = LOOK_PROMPT.format(obj=_article(obj), scale=f", one tick = {f} cells" if f > 1 else "",
                              colors=", ".join(COLORS))
    return loop.run_until_complete(_ask([photo, view], text, lambda t: parse_look(t, f)))


def paint(parts: list[dict], boxes: list[dict]) -> tuple[list[dict], int]:
    """Plates and bricks with a cell in a box take its color (glass and working parts keep theirs)."""
    out, n = [], 0
    for p in parts:
        q = p
        if p.get("type") in ("plate", "brick") and p.get("color") not in ("trans_clear", "trans_black"):
            for b in boxes:
                if any(b["x"][0] <= x <= b["x"][1] and b["y"][0] <= y <= b["y"][1] and b["z"][0] <= z <= b["z"][1]
                       for x, y, z in _cells(p)):
                    q = p | {"color": b["color"]}
            n += q is not p
        out.append(q)
    return out, n


def look(photo: Path, parts: list[dict], obj: str, work: Path, loop, *, cut: bool = True, allow_paint: bool = True,
         rounds: int = 2) -> tuple[list[dict], list[dict]]:
    """Phase 2: Spark's paint and glass punch list, carried out, up to `rounds` times. Returns (parts, log)."""
    from .spark_glass import apply as glaze_boxes

    log = []
    for rnd in range(1, rounds + 1):
        ans = ask_look(photo, parts, obj, work / f"look_{rnd}", loop)
        entry = {"round": rnd, "answer": ans}
        log.append(entry)
        if "error" in ans:
            break
        todo_paint = ans["paint"] if allow_paint else []
        if not todo_paint and not ans["glass"]:
            break
        if todo_paint:
            parts, entry["painted"] = paint(parts, todo_paint)
        if ans["glass"]:
            parts, rep = glaze_boxes(parts, {"glass": ans["glass"]}, cut=cut)
            entry["glazed"] = rep.get("glass_parts")
    return parts, log
