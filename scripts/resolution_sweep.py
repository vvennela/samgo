"""How much of the object survives at each resolution: the same SAM 3D mesh voxelized and built as a LEGO body at
several sizes (longest side in studs). Writes <out>/<name>_<studs>.json (parts), stats.json, and prints a line per size.

    PYTHONPATH=. python scripts/resolution_sweep.py <object.glb> <out dir> <name> 20 40 100 200
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from samgo.builder import build
from samgo.voxelize import load_mesh, voxelize


def main() -> None:
    glb, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    sizes = [int(s) for s in sys.argv[4:]]
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    mesh = load_mesh(glb)
    for studs in sizes:
        t = time.time()
        grid = voxelize(mesh, studs)
        tv = time.time() - t
        t = time.time()
        parts = build(grid, attempts=1, pull=lambda p: 0.0).pieces
        tb = time.time() - t
        (out / f"{name}_{studs}.json").write_text(json.dumps(parts))
        np.savez_compressed(out / f"{name}_{studs}_grid.npz", grid=grid)
        s = {"studs": studs, "grid": list(grid.shape), "cells": int((grid > 0).sum()), "parts": len(parts),
             "voxelize_s": round(tv, 1), "build_s": round(tb, 1)}
        stats[f"{name}_{studs}"] = s
        stats_path.write_text(json.dumps(stats, indent=1))
        print(name, s, flush=True)


if __name__ == "__main__":
    main()
