"""S6: photo -> verified build + instructions, no manual steps.

    uv run python -m samgo.pipeline photos/beetle.jpg --prompt car --car -o runs/s6/beetle

1. SAM 3.1 segments the object, SAM 3D reconstructs it (S1).
2. The mesh becomes a LEGO-scale target grid (S2).
3. For a vehicle: SAM 3.1 finds the wheels, and the car builder places a Technic chassis, axles and
   wheels there with the body on top (S4f); otherwise the structural builder fills the target (S4).
4. The verifiers check the result (S3a structure, S3b stability, S3c function for vehicles); failures are
   reported. Instructions are written when structure and function pass; stability is reported alongside.
5. Instructions are written (S5).
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from .builder import build
from .car import build_car
from .perceive import perceive
from .stability import stability
from .voxelize import SIZES, load_mesh, render, voxelize
from .wheels import detect


def run(photo: Path, prompt: str, out: Path, car: bool, size: str = "medium", reuse: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if not (reuse and (out / "object.glb").exists()):
        asyncio.run(perceive(photo, prompt, out))
    target = voxelize(load_mesh(out / "object.glb"), SIZES[size])
    np.savez_compressed(out / "target.npz", target=target)
    render(target, out / "target.png")

    summary: dict = {"photo": str(photo), "target_shape": list(target.shape), "cells": int((target > 0).sum())}
    if car:
        wheels = json.loads((out / "wheels.json").read_text()) if reuse and (out / "wheels.json").exists() \
            else asyncio.run(detect(out))
        result = build_car(target, wheel_fractions=tuple(wheels["fractions"]))
        if result is None:
            summary["error"] = "car builder refused the target (too wide for an axle, or no chassis split)"
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            return summary
        parts, structure, function = result.parts, result.structure, result.function
        summary |= {"wheels": wheels["source"], "wheel_fractions": wheels["fractions"],
                    "function_passed": function.passed, "function_issues": sorted(function.codes)}
    else:
        b = build(target)
        parts, structure = b.pieces, b.report
    summary |= {"parts": len(parts), "structure_passed": structure.passed,
                "structure_issues": sorted(structure.codes), "iou": round(structure.metrics.get("iou", 0.0), 3)}
    stands = stability(parts, vehicle=car)
    summary |= {"stands": stands.passed, "peak_pull_per_stud": None if stands.metrics.get("peak_tension") is None
                else round(4 * stands.metrics["peak_tension"], 1), "stability_issues": sorted(stands.codes)}
    (out / "build.json").write_text(json.dumps(parts))
    if summary["structure_passed"] and summary.get("function_passed", True):
        from .instructions import write  # matplotlib only when there is something to draw
        summary["instructions"] = str(write(parts, out / "instructions"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="photo -> verified LEGO-compatible build + instructions")
    ap.add_argument("photo", type=Path)
    ap.add_argument("--prompt", required=True, help="what to segment, e.g. 'car'")
    ap.add_argument("--car", action="store_true", help="build a rolling vehicle (chassis, axles, wheels)")
    ap.add_argument("--size", choices=SIZES, default="medium")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--fresh", action="store_true", help="redo perception even if outputs exist")
    args = ap.parse_args()
    print(json.dumps(run(args.photo, args.prompt, args.out, args.car, args.size, reuse=not args.fresh), indent=2))


if __name__ == "__main__":
    main()
