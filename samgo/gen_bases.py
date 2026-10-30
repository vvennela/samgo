"""Training bases: Objaverse vehicles turned into verified rolling LEGO cars (spec.md section 0, training data step 1).

Each vehicle mesh is voxelized (mirror-aligned) at several lengths and built by the arched car builder; only cars
that pass structure (S3a), function (S3c) and stability (S3b) are kept. Our photo cars are never used here: they
are the held-out test set.

    uv run python -m samgo.gen_bases --out runs/bases --sizes 16 20 24 --workers 8
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

VEHICLES = ("car_(automobile)", "police_cruiser", "race_car", "pickup_truck", "jeep", "minivan", "limousine",
            "ambulance", "school_bus", "bus_(vehicle)", "truck", "tow_truck", "fire_engine", "garbage_truck",
            "motor_vehicle")
MAX_MB = 40  # very large files (millions of triangles) are slow to load and voxelize; skipped


def _one(uid: str, category: str, path: str, sizes: list[int], out: str) -> list[dict]:
    from .arches import build_arched_car
    from .stability import stability
    from .voxelize import load_mesh, voxelize

    rows = []
    try:
        mesh = load_mesh(Path(path))
    except Exception as e:  # noqa: BLE001
        return [{"uid": uid, "category": category, "result": f"load error: {type(e).__name__}"}]
    for size in sizes:
        row = {"uid": uid, "category": category, "size": size}
        try:
            target = voxelize(mesh, size)
            car = build_arched_car(target, attempts=4)
            if car is None:
                row["result"] = "refused"
            else:
                stands = stability(car.parts, vehicle=True).passed
                ok = car.structure.passed and car.function.passed and stands
                row |= {"result": "pass" if ok else "fail", "parts": len(car.parts),
                        "wheels": [(w.edge, w.size) for w in car.wheels]}
                if ok:
                    name = f"{uid}_{size}"
                    np.savez_compressed(Path(out) / f"{name}.npz", target=car.target)
                    (Path(out) / f"{name}.json").write_text(json.dumps(car.parts))
                    row["file"] = name
        except Exception as e:  # noqa: BLE001
            row["result"] = f"error: {type(e).__name__}: {str(e)[:60]}"
        rows.append(row)
    return rows


def main() -> None:
    import objaverse
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/bases"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 20, 24])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N vehicles only (0 = all)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.jsonl"
    done = {json.loads(line)["uid"] for line in open(index_path)} if index_path.exists() else set()

    lvis = objaverse.load_lvis_annotations()
    paths = objaverse._load_object_paths()
    todo = [(u, c) for c in VEHICLES if c in lvis for u in lvis[c] if u not in done]
    todo = list(dict.fromkeys(todo))[: args.limit or None]
    print(f"{len(todo)} vehicles to build at sizes {args.sizes}", flush=True)

    def fetch(uid):
        try:
            p = hf_hub_download("allenai/objaverse", paths[uid], repo_type="dataset")
            return p if os.path.getsize(p) <= MAX_MB * 1e6 else None
        except Exception:  # noqa: BLE001
            return None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as dl, ProcessPoolExecutor(args.workers) as pool, index_path.open("a") as index:
        futures = []
        for (uid, cat), path in zip(todo, dl.map(lambda uc: fetch(uc[0]), todo)):
            if path is None:
                index.write(json.dumps({"uid": uid, "category": cat, "result": "skipped (download or size)"}) + "\n")
                continue
            futures.append(pool.submit(_one, uid, cat, path, args.sizes, str(args.out)))
        passed = total = 0
        for f in as_completed(futures):
            for row in f.result():
                index.write(json.dumps(row) + "\n")
                total += 1
                passed += row.get("result") == "pass"
            index.flush()
            if total % 30 < len(args.sizes):
                print(f"{passed}/{total} builds pass", flush=True)
    print(f"done: {passed}/{total} builds pass", flush=True)


if __name__ == "__main__":
    main()
