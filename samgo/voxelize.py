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
            mesh.visual = mesh.visual.to_color()
        parts.append(mesh)
    if not parts:
        raise ValueError(f"{path} has no triangle meshes")
    mesh = trimesh.util.concatenate(parts)
    if up == "y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif up != "z":
        raise ValueError(f"up must be 'y' or 'z', not {up!r}")
    return mesh


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


def voxelize(mesh: trimesh.Trimesh, longest_studs: int, seed: int = 0) -> np.ndarray:
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounds[0])
    scale = longest_studs / mesh.extents.max()
    mesh.apply_transform(np.diag([scale, scale, scale * PLATE_PER_STUD, 1.0]))
    shape = np.maximum(np.ceil(mesh.extents - 1e-9).astype(int), 1)

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
    return np.where(solid, surface_color[tuple(nearest)], 0).astype(np.int32)


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
