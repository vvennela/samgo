"""S5: building instructions. A verified build -> numbered steps a person can follow.

Plates and bricks go in bottom-up, one layer per step (big layers split into a few steps); axles,
then wheels, come last. Each step is a 3D picture: parts already placed are faded, new parts are
in full color with dark edges, and the step's parts are listed. Writes step_NN.png, an
instructions.md with the parts per step, and the full bill of materials.

    uv run python -m samgo.instructions build.json out/
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .parts import COLORS  # noqa: E402
from .technic import HOLE_HEIGHT, PLATE, STUD, WHEEL_DIAMETER  # noqa: E402

MAX_PER_STEP = 10
HEIGHT = {"plate": 1, "brick": 3, "technic_brick": 3}


def name(p: dict) -> str:
    color = p.get("color", "").replace("_", " ")
    if p["type"] == "axle":
        return f"Technic axle {p['length']}"
    if p["type"] == "wheel":
        return "wheel 30.4 x 14 with tire"
    a, b = sorted((p["w"], p["l"]))
    kind = {"technic_brick": f"Technic brick ({p.get('hole', 'round')} holes)"}.get(p["type"], p["type"])
    return f"{a}x{b} {kind}, {color}"


def steps(parts: list[dict]) -> list[list[dict]]:
    solid = sorted((p for p in parts if p["type"] in HEIGHT), key=lambda p: (p["z"], p["y"], p["x"]))
    out, layer = [], []
    for p in solid:
        if layer and (p["z"] != layer[0]["z"] or len(layer) == MAX_PER_STEP):
            out.append(layer)
            layer = []
        layer.append(p)
    out += [layer] if layer else []
    for kind in ("axle", "wheel"):
        if group := [p for p in parts if p["type"] == kind]:
            out.append(group)
    return out


def _box_faces(x0, x1, y0, y1, z0, z1):
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    return [c[[0, 1, 2, 3]], c[[4, 5, 6, 7]], c[[0, 1, 5, 4]], c[[2, 3, 7, 6]], c[[1, 2, 6, 5]], c[[0, 3, 7, 4]]]


def _draw(ax, p: dict, new: bool) -> None:
    alpha, edge = (1.0, (0.1, 0.1, 0.1, 0.9)) if new else (0.12, (0.4, 0.4, 0.4, 0.15))
    if p["type"] in HEIGHT:
        rgb = np.array(COLORS[p["color"]][1]) / 255
        x0, y0, z0 = p["x"] * STUD, p["y"] * STUD, p["z"] * PLATE
        faces = _box_faces(x0, x0 + p["w"] * STUD, y0, y0 + p["l"] * STUD, z0, z0 + HEIGHT[p["type"]] * PLATE)
        ax.add_collection3d(Poly3DCollection(faces, facecolor=(*rgb, alpha), edgecolor=edge, linewidth=0.4))
        return
    z = p["z"] * PLATE + HOLE_HEIGHT
    if p["type"] == "axle":
        edge_mm, start = (p["x"] * STUD, p["y"] * STUD) if p["axis"] == "y" else (p["y"] * STUD, p["x"] * STUD)
        span = [start, start + p["length"] * STUD]
        xs, ys = ([edge_mm] * 2, span) if p["axis"] == "y" else (span, [edge_mm] * 2)
        ax.plot(xs, ys, [z, z], color=(0.3, 0.3, 0.3, alpha), linewidth=3 if new else 1)
        return
    r, t = WHEEL_DIAMETER / 2, np.linspace(0, 2 * np.pi, 24)
    edge_mm, start = (p["x"] * STUD, p["y"] * STUD) if p["axis"] == "y" else (p["y"] * STUD, p["x"] * STUD)
    for s in (start, start + p["width"] * STUD):
        a, zc = edge_mm + r * np.cos(t), z + r * np.sin(t)
        xs, ys = (a, np.full_like(a, s)) if p["axis"] == "y" else (np.full_like(a, s), a)
        ax.add_collection3d(Poly3DCollection([np.c_[xs, ys, zc]], facecolor=(0.12, 0.12, 0.12, 0.9 * alpha + 0.05),
                                             edgecolor=edge, linewidth=0.4))


def render_step(parts_before: list[dict], new: list[dict], bounds, path: Path, title: str) -> None:
    fig = plt.figure(figsize=(6, 5), dpi=110)
    ax = fig.add_subplot(projection="3d")
    for p in parts_before:
        _draw(ax, p, False)
    for p in new:
        _draw(ax, p, True)
    (x0, x1), (y0, y1), (z0, z1) = bounds
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=28, azim=-60)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write(parts: list[dict], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    plan = steps(parts)
    r = WHEEL_DIAMETER / 2
    xs = [v for p in parts for v in ((p["x"], p["x"] + p.get("w", 0)) if p["type"] in HEIGHT else (p["x"], p["x"]))]
    ys = [v for p in parts for v in ((p["y"], p["y"] + p.get("l", 0)) if p["type"] in HEIGHT
                                     else (p["y"], p["y"] + p.get("length", p.get("width", 0))))]
    zs = [p["z"] + HEIGHT.get(p["type"], 0) for p in parts]
    has_wheels = any(p["type"] == "wheel" for p in parts)
    bounds = ((min(xs) * STUD - (r if has_wheels else 0), max(xs) * STUD + (r if has_wheels else 0)),
              (min(ys) * STUD, max(ys) * STUD),
              (min(0.0, HOLE_HEIGHT - r) if has_wheels else 0.0, max(zs) * PLATE))
    render_step([], parts, bounds, out / "finished.png", "Finished model")
    lines = ["# Building instructions", "", "![finished model](finished.png)", "",
             f"{len(parts)} parts in {len(plan)} steps.", "", "## Parts", ""]
    lines += [f"- {n} × {name_}" for name_, n in sorted(Counter(name(p) for p in parts).items())]
    done: list[dict] = []
    for i, group in enumerate(plan, 1):
        image = out / f"step_{i:02d}.png"
        render_step(done, group, bounds, image, f"Step {i} of {len(plan)}")
        lines += ["", f"## Step {i}", "", f"![step {i}]({image.name})", ""]
        lines += [f"- {n} × {name_}" for name_, n in sorted(Counter(name(p) for p in group).items())]
        done += group
    doc = out / "instructions.md"
    doc.write_text("\n".join(lines) + "\n")
    return doc


if __name__ == "__main__":
    print(write(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])))
