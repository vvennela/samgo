"""Offline checks for S1 mask handling and S2 voxelization, using synthetic meshes."""

import numpy as np
import pytest
import trimesh

from samgo.parts import COLOR_INDEX
from samgo.perceive import best_mask
from samgo.voxelize import load_mesh, nearest_color, render, voxelize


def colored_box(extents, rgb, center=(0, 0, 0)):
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    box.visual = trimesh.visual.ColorVisuals(box, vertex_colors=np.tile([*rgb, 255], (len(box.vertices), 1)))
    return box


# --- S1 mask choice -------------------------------------------------------------

def test_best_mask_prefers_confidence_and_scales_to_photo():
    small, big = np.zeros((4, 4), np.uint8), np.ones((4, 4), np.uint8)
    small[1:3, 1:3] = 1
    mask, confidence = best_mask([(big, 0.4), (small, 0.9)], size=(8, 12))
    assert confidence == 0.9 and mask.shape == (12, 8) and mask.sum() == 4 * 6


def test_best_mask_falls_back_to_largest():
    small, big = np.zeros((4, 4), np.uint8), np.ones((4, 4), np.uint8)
    small[0, 0] = 1
    mask, confidence = best_mask([(small, None), (big, None)], size=(4, 4))
    assert confidence is None and mask.all()


def test_best_mask_needs_a_match():
    with pytest.raises(RuntimeError):
        best_mask([], size=(4, 4))


# --- S2 voxelization ------------------------------------------------------------

@pytest.mark.parametrize("rgb,name", [((180, 0, 0), "red"), ((250, 200, 10), "yellow"),
                                       ((20, 30, 40), "black"), ((240, 240, 240), "white")])
def test_nearest_color(rgb, name):
    assert nearest_color(np.array(rgb)) == COLOR_INDEX[name]


def test_box_scales_to_longest_side_and_fills():
    # 2 x 1 x 1 box (z-up): longest side 20 studs -> 20 x 10 studs x 25 plates (10 studs / 0.4).
    target = voxelize(colored_box((2, 1, 1), (180, 0, 0)), longest_studs=20)
    assert target.shape == (20, 10, 25)
    assert (target > 0).mean() > 0.98  # solid, not a hollow shell
    assert set(np.unique(target)) == {COLOR_INDEX["red"]}


def test_tall_box_counts_height_in_studs():
    target = voxelize(colored_box((1, 1, 2), (180, 0, 0)), longest_studs=10)
    assert target.shape == (5, 5, 25)  # 10 studs tall = 25 plates


def test_two_colors_stay_on_their_sides():
    mesh = trimesh.util.concatenate([colored_box((1, 1, 1), (180, 0, 0), (0, 0, 0)),
                                     colored_box((1, 1, 1), (30, 90, 168), (1, 0, 0))])
    target = voxelize(mesh, longest_studs=10)
    assert target.shape[0] == 10
    assert (target[:4] == COLOR_INDEX["red"]).all() and (target[7:] == COLOR_INDEX["blue"]).all()
    # The red/blue edge must not average into a third, muddy color.
    assert set(np.unique(target)) == {COLOR_INDEX["red"], COLOR_INDEX["blue"]}


def test_stray_islands_are_dropped():
    mesh = trimesh.util.concatenate([colored_box((1, 1, 1), (180, 0, 0)),
                                     colored_box((0.05, 0.05, 0.05), (180, 0, 0), (2, 2, 2))])
    target = voxelize(mesh, longest_studs=10)
    labels_kept = (target > 0)
    assert labels_kept[: target.shape[0] // 2].sum() > 0 and not labels_kept[-1, -1, -1]


def test_glb_is_turned_z_up(tmp_path):
    # Tall along y in the file (glTF y-up) -> tall along z after loading.
    path = tmp_path / "tall.glb"
    colored_box((1, 3, 1), (180, 0, 0)).export(path)
    mesh = load_mesh(path)
    assert np.allclose(mesh.extents, (1, 1, 3))
    assert mesh.visual.vertex_colors[0, 0] == 180


def test_render_writes_png(tmp_path):
    target = voxelize(colored_box((2, 1, 1), (180, 0, 0)), longest_studs=12)
    render(target, tmp_path / "views.png")
    assert (tmp_path / "views.png").stat().st_size > 0
