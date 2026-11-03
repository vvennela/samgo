"""Spark reads a car's layout: where the wheels and the doors are, from the photo and a side render of the model.

Spark (the knowing model) sees the photo and a side view of the voxel model with numbered stud ticks along its
length and plate ticks up its height, and answers in the render's own coordinates, so nothing depends on which
way the photo faces:

    {"front_wheel_x": 4, "rear_wheel_x": 16, "doors": [{"x0": 7, "x1": 11, "z0": 4, "z1": 9}, ...]}

Code then uses these facts (axles at the wheel centres, each door cut inside its outline); every result is
still checked. Answers are cached per photo and model shape.
"""

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

PROMPT = """The first image is a photo of a car. The second is a side view of a LEGO-scale model of the same car, shaded by depth (lighter is nearer):
x runs along the car (numbered ticks, one per stud), z up its height (numbered ticks, one per plate).

Read where, on the side view, the real car's parts are. Answer with one JSON object and nothing else:
{"front_wheel_x": <x of the centre of the front wheel>, "rear_wheel_x": <x of the centre of the rear wheel>,
 "wheel_top_z": <z of the top of the wheels>,
 "doors": [{"x0": <x where the door begins>, "x1": <x where it ends>, "z0": <z of its bottom edge>,
            "z1": <z of its top edge, the bottom of its side window>}, ...]}

"doors" lists every side door on one side of the car, front to back, as many as the real car has on that side
(count them from the photo's shut lines). Use the photo to know which end is the front and where the shut lines
are; give positions on the side view's ticks."""


def render_side(target: np.ndarray, path: Path) -> None:
    """Side view (x along, z up) of the model with a tick per stud and per plate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    occ = target > 0
    X, Y, Z = target.shape
    img = np.ones((Z, X, 3))
    for x in range(X):
        for z in range(Z):
            ys = np.nonzero(occ[x, :, z])[0]
            if len(ys):  # shaded by depth: the nearest surface light, recessed ones darker (arches, sills, glass)
                depth = ys.min() / max(Y - 1, 1)
                img[z, x] = 0.85 - 0.6 * min(1.0, 2.5 * depth)
    fig, ax = plt.subplots(figsize=(max(6, X * 0.35), max(3, Z * 0.18)))
    ax.imshow(img, origin="lower", extent=(-0.5, X - 0.5, -0.5, Z - 0.5))
    ax.set_aspect(3.2 / 8.0)  # a plate is 3.2 mm, a stud 8 mm: true proportions
    ax.set_xticks(range(0, X, 2 if X > 24 else 1))
    ax.set_yticks(range(0, Z, 2 if Z > 20 else 1))
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_xlabel("x (studs)")
    ax.set_ylabel("z (plates)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def parse(text: str, shape: tuple[int, int, int]) -> dict | None:
    import re

    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    X, _, Z = shape
    try:
        out = {"front_wheel_x": int(d["front_wheel_x"]), "rear_wheel_x": int(d["rear_wheel_x"]),
               "wheel_top_z": int(d.get("wheel_top_z", 0))}
        doors = d.get("doors") or ([d["door"]] if d.get("door") else [])
        out["doors"] = [{k: int(door[k]) for k in ("x0", "x1", "z0", "z1")} for door in doors if door]
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= out["front_wheel_x"] < X and 0 <= out["rear_wheel_x"] < X):
        return None
    for dd in out["doors"]:
        dd["x0"], dd["x1"] = sorted((max(0, dd["x0"]), min(X - 1, dd["x1"])))
        dd["z0"], dd["z1"] = sorted((max(0, dd["z0"]), min(Z - 1, dd["z1"])))
    if out["doors"]:
        out["door"] = out["doors"][0]  # older readers: the front door
    return out


async def ask(photo: Path, target: np.ndarray, work: Path, client=None) -> dict:
    """Spark's layout for this car (cached per photo and model)."""
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    from .function_spec import BASE_URL, MODEL, _data_url

    work.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(Path(photo).read_bytes() + target.tobytes() + str(target.shape).encode() + PROMPT.encode()).hexdigest()[:16]
    cache = Path("runs/research_cache") / f"layout5_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    side = work / "side_view.png"
    render_side(target, side)
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    import os

    client = client or AsyncOpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=BASE_URL, max_retries=6)

    async def one():
        r = await client.chat.completions.create(
            model=MODEL, max_tokens=3000, extra_body={"reasoning_effort": "low"},
            messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                   {"type": "image_url", "image_url": {"url": _data_url(photo)}},
                                                   {"type": "image_url",
                                                    "image_url": {"url": _data_url(side, 1400)}}]}])
        return parse(r.choices[0].message.content or "", target.shape)

    # Self-consistency: several answers at once, then the consensus (Spark's single answers vary run to run).
    answers = [a for a in await asyncio.gather(*[one() for _ in range(SAMPLES)], return_exceptions=True)
               if isinstance(a, dict)]
    out = consensus(answers) if answers else {"error": "no parseable answer"}
    if "error" not in out:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out


SAMPLES = 5


def consensus(answers: list[dict]) -> dict:
    """Majority on which end is the front, then the median of each number among the answers that agree; the
    doors of the most common door count, field by field."""
    import statistics

    forward = [a for a in answers if a["front_wheel_x"] > a["rear_wheel_x"]]
    backward = [a for a in answers if a["front_wheel_x"] <= a["rear_wheel_x"]]
    agree = forward if len(forward) >= len(backward) else backward
    med = lambda k: int(round(statistics.median(a[k] for a in agree)))  # noqa: E731
    out = {"front_wheel_x": med("front_wheel_x"), "rear_wheel_x": med("rear_wheel_x"), "wheel_top_z": med("wheel_top_z")}
    counts = [len(a["doors"]) for a in agree]
    n = max(set(counts), key=counts.count)
    same = [sorted(a["doors"], key=lambda d: d["x0"]) for a in agree if len(a["doors"]) == n]  # same order in each
    out["doors"] = [{k: int(round(statistics.median(d[i][k] for d in same))) for k in ("x0", "x1", "z0", "z1")}
                    for i in range(n)]
    for a, b in zip(out["doors"], out["doors"][1:]):  # neighbours meet at a shut line, never overlap
        b["x0"] = max(b["x0"], a["x1"])
    if out["front_wheel_x"] > out["rear_wheel_x"]:  # front door first, as single answers list them
        out["doors"] = out["doors"][::-1]
    if out["doors"]:
        out["door"] = out["doors"][0]
    out["samples"] = len(answers)
    out["agreeing"] = len(agree)
    return out


def layout(photo: Path, target: np.ndarray, work: Path) -> dict:
    return asyncio.run(ask(photo, target, work))
