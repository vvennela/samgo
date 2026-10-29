"""StableText2Brick -> chat JSONL for Fireworks SFT (BrickGPT's prepare_finetuning_dataset, with a caption cap).

    uv run python -m brickgpt.make_sft --captions 2
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_from_disk

from .prompts import messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stabletext2brick/hf")
    ap.add_argument("--captions", type=int, default=2, help="captions per structure (BrickGPT used all 5)")
    ap.add_argument("--out", type=Path, default=Path("data/sft"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk(args.data)
    rows = [messages(c, r["bricks"]) for r in ds["train"] for c in rng.sample(r["captions"], args.captions)]
    rng.shuffle(rows)
    path = args.out / f"train_c{args.captions}.jsonl"
    with open(path, "w") as f:
        for m in rows:
            f.write(json.dumps({"messages": m}) + "\n")
    print(f"{path}: {len(rows)} rows, {path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
