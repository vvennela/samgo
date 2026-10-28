"""Fallback when vLLM can't load a LoRA adapter for this architecture: merge it into the base weights."""

import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel

from pilot.train import load_model


def main(base: str, adapter: str, out: str) -> None:
    # run.sh sets CUDA_VISIBLE_DEVICES= so this merges on CPU; the box has 384 GB of RAM.
    model = load_model(base)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    for f in Path(base).iterdir():  # tokenizer, chat template, processor configs
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json" \
                and not (Path(out) / f.name).exists():
            shutil.copy(f, Path(out) / f.name)


if __name__ == "__main__":
    main(*sys.argv[1:4])
