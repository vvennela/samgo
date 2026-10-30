"""S2: 3D mesh -> LEGO-scale target grid.

The target is an (X, Y, Z) int grid of COLOR_INDEX values (0 = empty) in the
verifier's units: x and y in studs, z in plates. A stud is 8 mm wide and a
plate 3.2 mm tall, so the mesh is stretched 2.5x vertically before sampling.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage

from .parts import COLOR_INDEX, COLORS

# Longest dimension of the model, in studs (height counted in studs too).
SIZES = {"small": 12, "medium": 20, "large": 32, "xlarge": 60}
PLATE_PER_STUD = 8.0 / 3.2
SAMPLES_PER_CELL_AREA = 16  # surface points per unit^2 of grid area, ~0.25-cell spacing


def load_mesh(path: Path, up: str = "y") -> trimesh.Trimesh:
    """Load a GLB as one vertex-colored, z-up mesh. glTF files are y-up."""
    scene = trimesh.load(path, force="scene")
    parts = []
    for mesh in scene.dump():
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        if not isinstance(mesh.visual, trimesh.visual.ColorVisuals):
            try:
                mesh.visual = mesh.visual.to_color()
            except (IndexError, ValueError, AttributeError):
                # Some downloaded models have texture coordinates that don't match their vertices; fall back to
                # the material's one color (base color, else the texture's average).
                mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=_material_color(mesh.visual))
        colors = np.asarray(mesh.visual.vertex_colors)
        if len(colors) != len(mesh.vertices):  # colors that don't line up with the vertices: one solid color
            solid = colors.reshape(-1, colors.shape[-1]).mean(axis=0) if colors.size else (150, 150, 150, 255)
            mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.asarray(solid, dtype=np.uint8))
        parts.append(mesh)
    if not parts:
        raise ValueError(f"{path} has no triangle meshes")
    mesh = trimesh.util.concatenate(parts)
    if up == "y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif up != "z":
        raise ValueError(f"up must be 'y' or 'z', not {up!r}")
    return mesh


def _material_color(visual) -> np.ndarray:
    """One RGBA color for a textured visual: the material's base color, else its texture's mean, else gray."""
    material = getattr(visual, "material", None)
    for attr in ("baseColorFactor", "main_color", "diffuse"):
        c = getattr(material, attr, None)
        if c is not None and len(np.ravel(c)) >= 3:
            c = np.asarray(c, dtype=float).ravel()
            c = c * 255 if c.max() <= 1.0 else c
            return np.append(c[:3], 255).astype(np.uint8)
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if image is not None:
        return np.append(np.asarray(image.convert("RGB")).reshape(-1, 3).mean(axis=0), 255).astype(np.uint8)
    return np.array([150, 150, 150, 255], dtype=np.uint8)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255) -> CIE Lab, so nearest-color matching follows perception."""
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


PALETTE_RGB = np.array([rgb for _, rgb in COLORS.values()], dtype=float)
PALETTE_LAB = to_lab(PALETTE_RGB)


def nearest_color(rgb: np.ndarray) -> np.ndarray:
    """Map (..., 3) sRGB colors to COLOR_INDEX values."""
    lab = to_lab(np.asarray(rgb, dtype=float))
    distance = ((lab[..., None, :] - PALETTE_LAB) ** 2).sum(-1)
    return distance.argmin(-1) + 1


MIRROR_ERROR = 0.02     # mean mirror mismatch below this share of the object's size counts as symmetric
SLICE_SYMMETRIC = 0.85  # a cross-section whose cells match their mirror this often is made exactly symmetric


def mirror_plane(mesh: trimesh.Trimesh, samples: int = 4000) -> tuple[float, float, float] | None:
    """The object's left-right mirror plane, as (angle about z in degrees, offset along the rotated y, error).

    Tries vertical planes along the long horizontal axis, turned by a few degrees and shifted a little,
    and keeps the one where mirrored surface points land closest to real ones. None if no plane fits within
    MIRROR_ERROR of the object's size (the object is not left-right symmetric).
    """
    from scipy.spatial import cKDTree

    pts, _ = trimesh.sample.sample_surface(mesh, samples, seed=0)
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(xy.T))
    base = np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1]))  # long axis
    base = (base + 90) % 180 - 90                    # never turn the object around front-to-back
    size = float(np.ptp(pts, axis=0).max())
    best = None
    for axis_turn in (0.0,):                         # left-right symmetry: the plane contains the long axis
        for turn in np.arange(-6, 7, 1.5):
            a = np.radians(base + axis_turn + turn)
            rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
            q = np.c_[xy @ rot.T, pts[:, 2]]
            tree = cKDTree(q)
            mid = (q[:, 1].min() + q[:, 1].max()) / 2
            for off in np.linspace(mid - 0.04 * size, mid + 0.04 * size, 9):
                m = q.copy()
                m[:, 1] = 2 * off - m[:, 1]
                err = tree.query(m)[0].mean() / size
                if best is None or err < best[2]:
                    best = (float(base + axis_turn + turn), float(off), float(err))
    if not best or best[2] >= MIRROR_ERROR:
        return None
    angle = best[0] if abs(best[0]) > 1.0 else 0.0   # snap a hair off square to square: no needless turning
    return angle, best[1], best[2]


def voxelize(mesh: trimesh.Trimesh, longest_studs: int, seed: int = 0, symmetric: bool = True) -> np.ndarray:
    """Mesh -> grid of COLOR_INDEX values (x, y studs; z plates).

    With `symmetric`, a left-right symmetric object is first turned so its mirror plane runs along x through the
    middle of the grid, and every cross-section that nearly matches its mirror is made exactly symmetric
    (one-sided features, like an open door, are kept as they are).
    """
    mesh = mesh.copy()
    plane = mirror_plane(mesh) if symmetric else None
    if plane is not None:
        angle, offset, _ = plane
        centre = mesh.bounds.mean(axis=0)
        mesh.apply_translation([-centre[0], -centre[1], 0])
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.radians(angle), [0, 0, 1]))
        mesh.apply_translation([0, -offset, 0])     # mirror plane at y = 0
        half = np.abs(mesh.bounds[:, 1]).max()
        mesh.apply_translation([0, half - (mesh.bounds[0, 1] + half), 0])  # y symmetric about 0
    mesh.apply_translation(-mesh.bounds[0])
    scale = longest_studs / mesh.extents.max()
    mesh.apply_transform(np.diag([scale, scale, scale * PLATE_PER_STUD, 1.0]))
    shape = np.maximum(np.ceil(mesh.extents - 1e-9).astype(int), 1)
    if plane is not None:  # the mirror plane (at the middle of y) exactly on the grid's middle (odd widths: the
        width = mesh.extents[1]                        # middle column mirrors onto itself)
        studs = max(int(np.ceil(width - 0.1)), 1)      # a hair over a whole stud still rounds down
        mesh.apply_translation([0, (studs - width) / 2, 0])
        shape[1] = studs

    count = int(np.clip(mesh.area * SAMPLES_PER_CELL_AREA, 20_000, 2_000_000))
    points, faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[faces], points)
    vertex_rgb = mesh.visual.vertex_colors[:, :3].astype(float)
    point_rgb = (bary[:, :, None] * vertex_rgb[mesh.faces[faces]]).sum(1)

    # Snap each sample to the palette, then let the samples in a cell vote.
    # Averaging RGB first would turn a red/blue edge into a muddy third color.
    cells = np.clip(np.floor(points).astype(int), 0, shape - 1)
    flat = np.ravel_multi_index(cells.T, shape)
    n = len(COLORS)
    votes = np.bincount(flat * n + nearest_color(point_rgb) - 1, minlength=np.prod(shape) * n).reshape(-1, n)
    surface = (votes.sum(1) > 0).reshape(shape)
    surface_color = (votes.argmax(1) + 1).reshape(shape)

    solid = ndimage.binary_fill_holes(surface)
    labels, n = ndimage.label(solid)
    if n > 1:  # drop stray islands; keep the object itself
        solid = labels == np.bincount(labels.ravel())[1:].argmax() + 1
    surface &= solid

    # Interior cells take the color of the nearest surface cell.
    _, nearest = ndimage.distance_transform_edt(~surface, return_indices=True)
    grid = np.where(solid, surface_color[tuple(nearest)], 0).astype(np.int32)
    if plane is not None:
        grid = _mirror_slices(grid, votes.reshape(*shape, len(COLORS)))
    return grid


def _mirror_slices(grid: np.ndarray, votes: np.ndarray) -> np.ndarray:
    """Make each cross-section along x that nearly matches its mirror (across the grid's middle) exactly
    symmetric: a cell is filled if it or its mirror is, and the pair takes the color with more surface votes."""
    out = grid.copy()
    mirrored = grid[:, ::-1, :]
    pair_votes = votes + votes[:, ::-1, :, :]
    pair_color = pair_votes.argmax(-1) + 1
    for x in range(grid.shape[0]):
        occ, mocc = grid[x] > 0, mirrored[x] > 0
        union = occ | mocc
        if union.any() and (occ & mocc).sum() / union.sum() >= SLICE_SYMMETRIC:
            fill = np.where(pair_votes[x].sum(-1) > 0, pair_color[x], np.maximum(grid[x], mirrored[x]))
            out[x] = np.where(union, fill, 0)
    return out


def render(target: np.ndarray, path: Path, cell: int = 10) -> None:
    """Front, side, and top orthographic views; farther cells are drawn darker."""
    rgb = np.vstack([[0, 0, 0], PALETTE_RGB]).astype(float)
    plate = cell * 3.2 / 8  # a plate is 0.4 studs tall

    def view(axis: int, flip: bool) -> np.ndarray:
        occ = target > 0
        if flip:
            occ, grid = np.flip(occ, axis), np.flip(target, axis)
        else:
            grid = target
        depth = occ.argmax(axis)
        hit = occ.any(axis)
        color = np.take_along_axis(grid, np.expand_dims(depth, axis), axis).squeeze(axis)
        shade = 1 - 0.6 * depth / max(target.shape[axis] - 1, 1)
        image = rgb[color] * shade[..., None]
        image[~hit] = 255
        return image.astype(np.uint8)

    front = view(1, False).transpose(1, 0, 2)[::-1]    # look along +y: x across, z up
    side = view(0, True).transpose(1, 0, 2)[::-1]      # look along -x: y across, z up
    top = view(2, True).transpose(1, 0, 2)[::-1]       # look down: x across, y up
    tiles = [("front", front, plate), ("side", side, plate), ("top", top, cell)]
    images = [(name, Image.fromarray(a).resize((a.shape[1] * cell, max(1, round(a.shape[0] * h))), Image.NEAREST))
              for name, a, h in tiles]

    pad = 20
    canvas = Image.new("RGB", (sum(i.width for _, i in images) + pad * 4, max(i.height for _, i in images) + pad * 2 + 14), "white")
    draw, x = ImageDraw.Draw(canvas), pad
    for name, image in images:
        canvas.paste(image, (x, pad + 14))
        draw.text((x, pad - 4), name, fill="black")
        x += image.width + pad
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="GLB mesh -> LEGO-scale target grid")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--size", choices=SIZES, default="medium")
    parser.add_argument("--up", choices=["y", "z"], default="y", help="mesh up axis (glTF is y-up)")
    parser.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: next to the mesh)")
    args = parser.parse_args()

    out = args.out or args.mesh.parent
    out.mkdir(parents=True, exist_ok=True)
    target = voxelize(load_mesh(args.mesh, args.up), SIZES[args.size])
    name = f"target-{args.size}"
    np.savez_compressed(out / f"{name}.npz", target=target)
    render(target, out / f"{name}.png")
    colors = {c: int((target == i).sum()) for c, i in COLOR_INDEX.items() if (target == i).any()}
    print(json.dumps({"shape": list(target.shape), "cells": int((target > 0).sum()), "colors": colors,
                      "files": [str(out / f"{name}.npz"), str(out / f"{name}.png")]}, indent=2))


if __name__ == "__main__":
    main()
