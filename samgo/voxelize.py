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
ONE_SIDED = 0.03        # a connected one-sided region this big (share of the object's cells) is a real feature
SMOOTH_IOU = 0.95       # shape smoothing: the largest blur that keeps this much overlap with the raw shape
COLOR_KEEP = 0.90       # color smoothing: the strongest smoothing that keeps this share of cells on their color
SIGMAS_MM = (2.0, 4.0, 6.0, 8.0)
LAMBDAS = (0.3, 0.6, 1.0, 1.5)


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
    grid = smooth_shape(grid)
    votes = votes.reshape(*shape, len(COLORS))
    if plane is not None:
        grid = _mirror(grid, votes)
    return merge_small_regions(smooth_colors(grid, mirrored=plane is not None))


CELL_MM = (8.0, 8.0, 3.2)


def smooth_shape(grid: np.ndarray) -> np.ndarray:
    """Blur the shape's signed distance (in mm, so plates and studs weigh right) and re-threshold, with the largest
    blur (SIGMAS_MM) that keeps IoU with the raw shape >= SMOOTH_IOU. New cells take the nearest cell's color."""
    occ = grid > 0
    if not occ.any():
        return grid
    sdf = ndimage.distance_transform_edt(~occ, sampling=CELL_MM) - ndimage.distance_transform_edt(occ, sampling=CELL_MM)
    best = occ
    for sigma in SIGMAS_MM:
        new = ndimage.gaussian_filter(sdf, [sigma / c for c in CELL_MM]) < 0
        if (new & occ).sum() / max((new | occ).sum(), 1) < SMOOTH_IOU:
            break
        best = new
    if best is occ:
        return grid
    best = best.copy()
    best[:, :, :3] = occ[:, :, :3]  # the bottom brick layer as voxelized: never bridge the gaps between tires/feet
    _, nearest = ndimage.distance_transform_edt(~occ, return_indices=True)
    return np.where(best, grid[tuple(nearest)], 0).astype(np.int32)


def _mirror(grid: np.ndarray, votes: np.ndarray) -> np.ndarray:
    """Make the grid symmetric across the middle of y, except connected one-sided regions of at least ONE_SIDED of
    the object (an open door, a spare wheel), which stay as they are. Pairs take the color with more votes."""
    occ = grid > 0
    mocc = occ[:, ::-1, :]
    one_sided = occ ^ mocc
    labels, n = ndimage.label(one_sided)
    keep = np.zeros_like(occ)
    if n:
        sizes = ndimage.sum(one_sided, labels, range(1, n + 1))
        for i, size in enumerate(sizes, 1):
            if size >= ONE_SIDED * occ.sum():
                keep |= labels == i
    keep = keep | keep[:, ::-1, :]                      # the region and the empty space it faces stay as is
    pair_votes = votes + votes[:, ::-1, :, :]
    pair_color = np.where(pair_votes.sum(-1) > 0, pair_votes.argmax(-1) + 1, np.maximum(grid, grid[:, ::-1, :]))
    sym = np.where(occ | mocc, pair_color, 0)
    return np.where(keep, grid, sym).astype(np.int32)


def smooth_colors(grid: np.ndarray, mirrored: bool) -> np.ndarray:
    """Potts smoothing: each cell takes the color minimizing [differs from its photo color] + lambda x (face
    neighbours of another color), by repeated local updates. The largest lambda (LAMBDAS) keeping COLOR_KEEP of
    cells on their photo color wins. With `mirrored`, mirror pairs are decided together, so it stays symmetric."""
    occ = grid > 0
    if not occ.any():
        return grid
    k = len(COLORS)
    best = grid
    for lam in LAMBDAS:
        cur = grid.copy()
        for _ in range(4):
            counts = np.zeros(grid.shape + (k,))
            pad = np.pad(cur, 1)
            for axis in range(3):
                for shift in (1, -1):
                    nb = np.roll(pad, shift, axis=axis)[1:-1, 1:-1, 1:-1]
                    for c in range(1, k + 1):
                        counts[..., c - 1] += nb == c
            cost = lam * (counts.sum(-1, keepdims=True) - counts)          # neighbours of another color
            cost += (np.arange(1, k + 1) != grid[..., None])               # leaving the photo color
            if mirrored:
                cost = cost + cost[:, ::-1, :, :]
            cur = np.where(occ, cost.argmin(-1) + 1, 0)
        if (cur[occ] == grid[occ]).mean() < COLOR_KEEP:
            break
        best = cur
    return best.astype(np.int32)


def merge_small_regions(grid: np.ndarray, min_share: float = 0.02) -> np.ndarray:
    """Even colors: every connected patch of one color smaller than `min_share` of the model (at least 8 cells)
    takes the most common color around it. Photo reflections and shadows leave scattered patches that Potts
    smoothing keeps; large regions (the body color, windows, lights) stay."""
    from scipy import ndimage

    grid = grid.copy()
    occ = grid > 0
    small = max(8, int(min_share * occ.sum()))
    for _ in range(5):
        changed = False
        for c in np.unique(grid[occ]):
            labels, n = ndimage.label(grid == c)
            if n < 2 and (grid == c).sum() >= small:
                continue
            sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
            for k in np.nonzero(sizes < small)[0] + 1:
                patch = labels == k
                ring = ndimage.binary_dilation(patch) & ~patch & occ
                around = grid[ring]
                around = around[around != c]
                if around.size:
                    grid[patch] = np.bincount(around).argmax()
                    changed = True
        if not changed:
            break
    return grid


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
