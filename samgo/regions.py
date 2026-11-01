"""Generic part regions of a voxel model, and a numbered picture of them for a vision model to name.

Nothing here knows what the object is: the thick core is what survives a morphological opening; everything else
splits into connected appendages (on an airliner: wings, tailplanes, fin, engines; on a chair: legs, back). Each
region gets a number, drawn at its centre in a top view and a side view, so Spark can say which region is which
("3 is the left wing") without coordinates.
"""

import numpy as np
from scipy import ndimage


def split(target: np.ndarray, core_radius: int = 1, min_cells: int = 6) -> np.ndarray:
    """Region labels (0 empty, 1 the core, 2.. appendages) over the voxel grid."""
    occ = target > 0
    ball = ndimage.generate_binary_structure(3, 1)
    core = ndimage.binary_opening(occ, structure=ball, iterations=core_radius)
    labels, n = ndimage.label(core)
    if n:
        core = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)  # the largest thick body
    rest = occ & ~core
    parts, m = ndimage.label(rest)
    out = np.where(core, 1, 0)
    k = 2
    for i in range(1, m + 1):
        cells = parts == i
        if cells.sum() >= min_cells:
            out[cells] = k
            k += 1
        else:
            out[cells] = 1  # specks belong to the core
    return out


def render(labels: np.ndarray, path) -> dict[int, tuple[float, float, float]]:
    """A top view and a side view with each region coloured and numbered; returns region centres (x, y, z)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = [int(i) for i in np.unique(labels) if i > 0]
    centres = {i: tuple(float(v) for v in np.argwhere(labels == i).mean(axis=0)) for i in ids}
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (axis, (u, v), title) in zip(axes, ((2, (0, 1), "top view (x right = along the body, y up)"),
                                                (1, (0, 2), "side view (x right, z up)"))):
        img = np.zeros(labels.shape[u:u + 1] + labels.shape[v:v + 1] + (3,)) + 1.0
        proj = np.zeros((labels.shape[u], labels.shape[v]), int)
        order = range(labels.shape[axis]) if axis == 2 else range(labels.shape[axis])
        for d in order:  # paint from back to front so the nearest region shows
            sl = [slice(None)] * 3
            sl[axis] = d
            layer = labels[tuple(sl)]
            proj = np.where(layer > 0, layer, proj)
        for i in ids:
            img[proj == i] = cmap((i - 1) % 20)[:3]
        ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower")
        for i, c in centres.items():
            ax.text(c[u], c[v], str(i), ha="center", va="center", fontsize=11, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.5))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return centres
