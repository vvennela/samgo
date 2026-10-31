"""S5: building instructions. A verified build -> numbered steps a person can follow.

Plates and bricks go in bottom-up, one layer per step (big layers split into a few steps); axles,
then wheels and retainers, come last. These are schematic groups, not an insertion-path planner.
Each step is a 3D picture: parts already placed are faded, new parts are
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
from .parts import parse_piece  # noqa: E402
from .technic import AXLE_RADIUS, DEFAULT_WHEEL, PLATE, STUD, TUBE_RADIUS, WHEELS, _line, _validate  # noqa: E402

MAX_PER_STEP = 10
HEIGHT = {"plate": 1, "brick": 3, "technic_brick": 3}
ATTACHMENTS = {"wheel", "bush", "half_bush"}
SUPPORTED = HEIGHT.keys() | ATTACHMENTS | {"axle"}


def _validate_parts(parts: list[dict]) -> None:
    if not isinstance(parts, list) or not parts:
        raise ValueError("instructions need a non-empty list of parts")
    for i, p in enumerate(parts):
        if not isinstance(p, dict) or not isinstance(p.get("type"), str) or p["type"] not in SUPPORTED:
            raise ValueError(f"unsupported instruction part {i}: {p!r}")
        try:
            if p["type"] in HEIGHT:
                parse_piece(p | {"type": "brick"} if p["type"] == "technic_brick" else p)
            if p["type"] not in ("plate", "brick") and (err := _validate(p)):
                raise ValueError(err)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"invalid instruction part {i}: {e}") from e


def name(p: dict) -> str:
    color = p.get("color", "").replace("_", " ")
    if p["type"] == "axle":
        return f"Technic {'stop ' if p.get('stop') else ''}axle {p['length']}"
    if p["type"] == "wheel":
        return f"wheel {p.get('size', DEFAULT_WHEEL).replace('x', ' x ')} with tire"
    if p["type"] in ("bush", "half_bush"):
        return "Technic bush (3713)" if p["type"] == "bush" else "Technic half bush (32123a)"
    if p["type"] not in HEIGHT:
        raise ValueError(f"unsupported instruction part: {p['type']}")
    a, b = sorted((p["w"], p["l"]))
    kind = {"technic_brick": f"Technic brick ({p.get('hole', 'round')} holes)"}.get(p["type"], p["type"])
    return f"{a}x{b} {kind}, {color}"


def steps(parts: list[dict]) -> list[list[dict]]:
    _validate_parts(parts)
    solid = sorted((p for p in parts if p["type"] in HEIGHT), key=lambda p: (p["z"], p["y"], p["x"]))
    out, layer = [], []
    for p in solid:
        if layer and (p["z"] != layer[0]["z"] or len(layer) == MAX_PER_STEP):
            out.append(layer)
            layer = []
        layer.append(p)
    out += [layer] if layer else []
    for kinds in ({"axle"}, ATTACHMENTS):
        if group := [p for p in parts if p["type"] in kinds]:
            out.append(group)
    return out


def _radius(p: dict) -> float:
    return (WHEELS[p.get("size", DEFAULT_WHEEL)][0] / 2 if p["type"] == "wheel"
            else AXLE_RADIUS if p["type"] == "axle" else TUBE_RADIUS)


def _bounds(parts: list[dict]) -> tuple:
    """Exact display envelopes in mm, including each wheel's size, position and orientation."""
    boxes = []
    for p in parts:
        if p["type"] in HEIGHT:
            lo = np.array([p['x'] * STUD, p['y'] * STUD, p['z'] * PLATE])
            hi = lo + [p['w'] * STUD, p['l'] * STUD, HEIGHT[p['type']] * PLATE]
        else:
            axis, edge, z, start, end = _line(p)
            r = _radius(p)
            lo, hi = ([edge - r, start, z - r], [edge + r, end, z + r]) if axis == 'y' else (
                [start, edge - r, z - r], [end, edge + r, z + r])
        boxes.append((lo, hi))
    low = np.min([b[0] for b in boxes], axis=0)
    high = np.max([b[1] for b in boxes], axis=0)
    padding = np.maximum((high - low) * 0.03, 1.0)
    return tuple(zip(low - padding, high + padding))


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
    axis, edge_mm, z, start, end = _line(p)
    if p["type"] == "axle":
        span = [start, end]
        xs, ys = ([edge_mm] * 2, span) if axis == "y" else (span, [edge_mm] * 2)
        ax.plot(xs, ys, [z, z], color=(0.3, 0.3, 0.3, alpha), linewidth=3 if new else 1)
        return
    r, t = _radius(p), np.linspace(0, 2 * np.pi, 25)

    def ring(s, radius):
        a, zc = edge_mm + radius * np.cos(t), z + radius * np.sin(t)
        xs, ys = (a, np.full_like(a, s)) if axis == "y" else (np.full_like(a, s), a)
        return np.c_[xs, ys, zc]

    front, back = ring(start, r), ring(end, r)
    faces = [[front[k], front[k + 1], back[k + 1], back[k]] for k in range(len(t) - 1)]
    if p["type"] == "wheel":
        faces += [front, back]
        rgb = (0.12, 0.12, 0.12)
    else:  # Retainers are collars with an axle hole, not wheels or unrendered parts.
        for s, outer in ((start, front), (end, back)):
            inner = ring(s, AXLE_RADIUS)
            faces += [[outer[k], outer[k + 1], inner[k + 1], inner[k]] for k in range(len(t) - 1)]
        rgb = (0.6, 0.6, 0.6)
    ax.add_collection3d(Poly3DCollection(faces, facecolor=(*rgb, alpha), edgecolor=edge, linewidth=0.4))


def render_step(parts_before: list[dict], new: list[dict], bounds, path: Path, title: str) -> None:
    fig = plt.figure(figsize=(6, 5), dpi=110)
    ax = fig.add_subplot(projection="3d")
    # Every step uses known assembly bounds. Set them before adding artists so partial
    # collections cannot trigger autoscaling on empty/degenerate intermediate extents.
    (x0, x1), (y0, y1), (z0, z1) = bounds
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_zlim(z0, z1)
    ax.set_autoscale_on(False)
    for p in parts_before:
        _draw(ax, p, False)
    for p in new:
        _draw(ax, p, True)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=28, azim=-60)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write(parts: list[dict], out: Path, *, certificate=None, certificate_policy=None) -> Path:
    if certificate is not None:
        from .certificates import DEFAULT_POLICY, check_certificate
        checked = check_certificate(parts, certificate, policy=certificate_policy or DEFAULT_POLICY)
        if not checked['passed']:
            raise ValueError('instruction certificate rejected: ' + '; '.join(checked['issues']))
        # Preserve the actual proved sequence; regrouping by height can invalidate prefix equilibrium.
        plan = [[parts[i]] for i in certificate['order']]
    else:
        plan = steps(parts)
    bounds = _bounds(parts)
    out.mkdir(parents=True, exist_ok=True)
    render_step([], parts, bounds, out / "finished.png", "Finished model")
    assurance = (
        'This exact one-part-at-a-time order has an independently checked certificate for vertical '
        'insertion and gravity equilibrium at every resting step under the declared idealized model. '
        'It uses no temporary supports in that model. Real connector insertion forces, hand/gripper '
        'access, tolerances and material strength remain unvalidated.'
        if certificate is not None else
        'These steps group parts by height, then axles, then wheels and retainers. '
        'Insertion access and temporary support have not been checked.'
    )
    lines = ["# Building instructions", "", "![finished model](finished.png)", "",
             f"{len(parts)} parts in {len(plan)} steps.", "", assurance, "", "## Parts", ""]
    if certificate is not None:
        (out / 'assembly_certificate.json').write_text(json.dumps(
            {'certificate': certificate, 'specification': checked['specification']}, indent=2) + '\n')
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
