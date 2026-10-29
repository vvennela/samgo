"""3D picture of a v2 build: boxes for bricks, plates and tiles; wedges for slopes; cylinders for round parts.

    uv run python -m brickgpt.render data/v2/train_structures.jsonl 0 out.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from samgo.palette import Part, parse  # noqa: E402

STUD, PLATE = 1.0, 0.4  # plot units: a plate is 0.4 studs tall
COLORS = {"brick": "#b9bec4", "plate": "#9aa3ad", "tile": "#e7e9ec", "slope": "#d4553b", "invslope": "#a33b28",
          "curve": "#e8913a", "roundbrick": "#3d7cc9", "roundplate": "#6aa0e0", "arch": "#3aa66b"}


def _box(x0, x1, y0, y1, z0, z1):
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    return [c[[0, 1, 2, 3]], c[[4, 5, 6, 7]], c[[0, 1, 5, 4]], c[[2, 3, 7, 6]], c[[1, 2, 6, 5]], c[[0, 3, 7, 4]]]


def _wedge(p: Part, z0, z1, inverted=False):
    """A slope: full height over the high row, falling to the bottom (or top, inverted) at the far edge."""
    x0, x1, y0, y1 = p.x, p.x + p.w, p.y, p.y + p.l
    low = z1 if inverted else z0
    # Corner heights: high along the back edge, low along the front edge.
    def h(x, y):
        front = {"+x": x == x1, "-x": x == x0, "+y": y == y1, "-y": y == y0}[p.facing]
        return low if front else (z1 if not inverted else z0)
    xs, ys = (x0, x1), (y0, y1)
    top = [[x, y, (h(x, y) if not inverted else z1)] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    bottom = [[x, y, (z0 if not inverted else h(x, y))] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    faces = [np.array(top), np.array(bottom)]
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        faces.append(np.array([bottom[a], bottom[b], top[b], top[a]]))
    return faces


def _cylinder(p: Part, z0, z1, n=16):
    r = p.w / 2 * 0.95
    cx, cy = p.x + p.w / 2, p.y + p.l / 2
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]
    faces = [np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)]]
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        faces.append(np.array([[*a, z0], [*b, z0], [*b, z1], [*a, z1]]))
    return faces


def faces(p: Part):
    z0, z1 = p.z * PLATE, (p.z + p.h) * PLATE
    if p.kind in ("slope", "curve"):
        return _wedge(p, z0, z1)
    if p.kind == "invslope":
        return _wedge(p, z0, z1, inverted=True)
    if p.kind.startswith("round"):
        return _cylinder(p, z0, z1)
    return _box(p.x, p.x + p.w, p.y, p.y + p.l, z0, z1)


def draw(parts: list[Part], path: str, title: str = "", highlight: set | None = None,
         colors: list[str] | None = None) -> None:
    """`colors`: one matplotlib color per part (e.g. painted from the photo); default colors by part kind."""
    fig = plt.figure(figsize=(6, 6), dpi=110)
    ax = fig.add_subplot(projection="3d")
    for i, p in enumerate(parts):
        alpha = 1.0 if highlight is None or i in highlight else 0.15
        ax.add_collection3d(Poly3DCollection(faces(p), facecolor=colors[i] if colors else COLORS[p.kind], alpha=alpha,
                                             edgecolor=(0.15, 0.15, 0.15, 0.5 * alpha), linewidth=0.3))
    xs = [v for p in parts for v in (p.x, p.x + p.w)]
    ys = [v for p in parts for v in (p.y, p.y + p.l)]
    zs = [(p.z + p.h) * PLATE for p in parts]
    ax.set_xlim(min(xs), max(xs)), ax.set_ylim(min(ys), max(ys)), ax.set_zlim(0, max(zs))
    ax.set_box_aspect((max(xs) - min(xs), max(ys) - min(ys), max(zs)))
    ax.view_init(elev=25, azim=-55)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = [json.loads(line) for line in open(sys.argv[1])]
    row = rows[int(sys.argv[2])]
    parts = [parse(line) for line in row["lines"].splitlines()]
    draw(parts, sys.argv[3], f"{row['structure_id']}  {row['kinds']}")
