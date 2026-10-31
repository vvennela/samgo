"""Compatibility CLI for learned running gear; uses the photo pipeline's assembly orchestrator.

    python -m samgo.demo_running_gear runs/s6/camry --model <sampler path> --label r3 --sequential
"""

import argparse
from pathlib import Path

from .arches import build_arched_car
from .assembly import Config, FireworksSampler, assemble, save
from .voxelize import load_mesh, voxelize

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('folders', type=Path, nargs='+')
    ap.add_argument('--model', default=None)
    ap.add_argument('--label', default='base')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--max-tokens', type=int, default=8000)
    ap.add_argument('--timeout', type=float, default=120)
    ap.add_argument('--sequential', action='store_true')
    args = ap.parse_args()
    sampler = FireworksSampler(args.model)
    config = Config(args.rounds, args.max_tokens, args.timeout, args.sequential)
    passed = True
    for folder in args.folders:
        folder = folder if folder.is_absolute() else REPO / folder
        car = build_arched_car(voxelize(load_mesh(folder / 'object.glb'), 20), attempts=6)
        if car is None:
            raise ValueError(f'could not construct a supported car from {folder}')
        label = args.label + ('_seq' if args.sequential else '')
        out = REPO / 'runs/demo' / folder.name
        parts, trace = assemble(car.parts, car.target, out / f'{label}_assembly', sampler, config)
        save(out / f'{label}.json', parts)
        save(out / f'{label}_log.json', trace)
        passed = passed and trace['passed']
        print(f"{folder.name}: {trace['status']}; trace: {out / f'{label}_assembly/trace.json'}", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
