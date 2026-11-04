"""The SAM 3D mesh in the 3D viewer's coordinates (mm, y up), placed exactly where samgo.voxelize put it on the stud
grid, so the greedy body's plates fill it in place (scripts/film_tiles.py mesh=...). Decimated by vertex clustering.

    PYTHONPATH=. python scripts/film_mesh.py object.glb longest_studs out.json [cell_studs]
"""
import base64
import json
import sys

import numpy as np
import trimesh

from samgo.technic import PLATE, STUD
from samgo.voxelize import PLATE_PER_STUD, load_mesh, mirror_plane


def to_grid(mesh: trimesh.Trimesh, longest_studs: int) -> trimesh.Trimesh:
    """samgo.voxelize.voxelize's placement: mirror plane, then scaled to studs (x, y) and plates (z)."""
    mesh = mesh.copy()
    plane = mirror_plane(mesh)
    if plane is not None:
        angle, offset, _ = plane
        centre = mesh.bounds.mean(axis=0)
        mesh.apply_translation([-centre[0], -centre[1], 0])
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.radians(angle), [0, 0, 1]))
        mesh.apply_translation([0, -offset, 0])
        half = np.abs(mesh.bounds[:, 1]).max()
        mesh.apply_translation([0, half - (mesh.bounds[0, 1] + half), 0])
    mesh.apply_translation(-mesh.bounds[0])
    scale = longest_studs / mesh.extents.max()
    mesh.apply_transform(np.diag([scale, scale, scale * PLATE_PER_STUD, 1.0]))
    if plane is not None:
        width = mesh.extents[1]
        studs = max(int(np.ceil(width - 0.1)), 1)
        mesh.apply_translation([0, (studs - width) / 2, 0])
    return mesh


def cluster(v: np.ndarray, f: np.ndarray, c: np.ndarray, cell: float):
    key = np.floor(v / cell).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = len(uniq)
    pos = np.zeros((n, 3)); col = np.zeros((n, 3)); cnt = np.zeros(n)
    np.add.at(pos, inv, v); np.add.at(col, inv, c); np.add.at(cnt, inv, 1)
    pos /= cnt[:, None]; col /= cnt[:, None]
    nf = inv[f]
    keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = np.unique(np.sort(nf[keep], axis=1), axis=0, return_index=True)[1]
    return pos, inv[f][keep][np.sort(nf)], col


def main():
    glb, longest, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    cell = float(sys.argv[4]) if len(sys.argv) > 4 else 0.18
    mesh = to_grid(load_mesh(glb), longest)
    v = np.asarray(mesh.vertices)          # x, y studs; z plates
    c = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(float) / 255
    grid_cell = np.array([cell, cell, cell * PLATE_PER_STUD])
    pos, faces, col = cluster(v / grid_cell * cell, np.asarray(mesh.faces), c, cell)
    pos = pos / cell * grid_cell
    page = np.stack([pos[:, 0] * STUD, pos[:, 2] * PLATE, pos[:, 1] * STUD], axis=1).astype(np.float32)  # viewer: x, up, y
    enc = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    json.dump({"pos": enc(page), "col": enc(col.astype(np.float32)), "idx": enc(faces.astype(np.uint32)),
               "n": int(len(page)), "faces": int(len(faces))}, open(out, "w"))
    print(f"{len(v)} -> {len(page)} vertices, {len(faces)} faces -> {out}")


if __name__ == "__main__":
    main()
