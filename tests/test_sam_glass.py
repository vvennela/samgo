from pathlib import Path

import numpy as np
import pytest

MESH = Path("runs/glass_proto/marked.glb")


@pytest.mark.skipif(not MESH.exists(), reason="needs the SAM 3D mesh of a window-marked photo")
def test_glass_marked_on_the_photo_comes_back_as_cells_on_both_sides_not_on_the_roof():
    from samgo.builder import build
    from samgo.perceive import GLASS_MARK
    from samgo.spark_glass import glaze_cells
    from samgo.voxelize import load_mesh, voxelize

    grid, mask = voxelize(load_mesh(MESH), 20, exact_mirror=True, marker=GLASS_MARK)
    assert 40 < mask.sum() < 0.15 * (grid > 0).sum()
    assert np.array_equal(mask, mask[:, ::-1, :])  # the hidden side gets the same windows
    top = {(x, y): z for x, y, z in np.argwhere(grid > 0)}
    roof = [(x, y) for x, y, z in np.argwhere(mask) if top.get((x, y)) == z and 3 <= y <= grid.shape[1] - 4]
    assert len(roof) < 0.25 * mask.sum()  # the middle of the roof stays body
    parts = build(grid, attempts=1, pull=lambda p: 0.0).pieces
    out, rep = glaze_cells(parts, {tuple(c) for c in np.argwhere(mask)})
    assert rep["glass_parts"] > 0 and any(p["color"] == "trans_clear" for p in out)
