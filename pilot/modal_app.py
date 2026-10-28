"""Pilot phase 2 on Modal: same job script as Baseten, on 2xH100.

    modal run --detach pilot/modal_app.py     # download (CPU, cached in a volume), then the GPU job
    modal volume get samgo-runs pilot-002 runs/pilot/pilot-002/modal

The model lives in the samgo-models volume, so a retry skips the 60 GB download. Results go to
the samgo-runs volume and are committed once a minute, so a crash keeps every finished step.
"""

import os
import subprocess
import threading
from pathlib import Path

import modal

RUN = "pilot-002"
ROOT = Path(__file__).resolve().parent.parent

image = (
    # A CUDA devel image, not debian_slim: vLLM's FlashInfer JIT-compiles kernels and needs nvcc (torch is cu130).
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("procps")
    .pip_install("vllm==0.32.0", "transformers==5.19.0", "peft==0.21.2", "accelerate", "hf_transfer",
                 "huggingface_hub[cli]", "numpy", "scipy")
    .add_local_dir(ROOT / "samgo", "/code/samgo", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "pilot", "/code/pilot", ignore=["**/__pycache__"])
    .add_local_file(ROOT / "runs/pilot/tasks/test_sub.jsonl", "/code/tasks/test_sub.jsonl")
    .add_local_file(ROOT / "runs/pilot/tasks/train.jsonl", "/code/tasks/train.jsonl")
    .add_local_file(ROOT / f"runs/pilot/{RUN}/aws/plan.json", "/code/aws_plan.json")
)
models = modal.Volume.from_name("samgo-models", create_if_missing=True)
runs = modal.Volume.from_name("samgo-runs", create_if_missing=True)
app = modal.App("samgo-pilot", image=image)


@app.function(cpu=2, timeout=10 * 60)
def check() -> None:
    """CPU-only image check (no GPU spend): nvcc present and the pinned stack imports."""
    subprocess.run(["bash", "-c", "nvcc --version | tail -1; python -c 'import vllm, torch, transformers, peft; "
                    "print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, \"cuda\", torch.version.cuda)'"],
                   check=True)


@app.function(volumes={"/models": models}, cpu=8, memory=16 * 1024, timeout=60 * 60)
def download() -> str:
    path = Path("/models/glimmer")
    if not (path / "config.json").exists():
        subprocess.run(["hf", "download", "meta-models/Muse-Glimmer-30B", "--local-dir", str(path)], check=True,
                       env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        models.commit()
    size = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.strip()
    print(f"[job] model ready: {size}", flush=True)
    return size


@app.function(gpu="H100:2", volumes={"/models": models, "/runs": runs}, cpu=16, memory=200 * 1024,
              timeout=170 * 60)
def job() -> int:
    stop = threading.Event()

    def committer():  # results survive a crash or timeout
        while not stop.wait(60):
            runs.commit()

    threading.Thread(target=committer, daemon=True).start()
    env = {**os.environ, "SKIP_SETUP": "1", "MODEL": "/models/glimmer", "SAVE": f"/runs/{RUN}", "BT_NUM_GPUS": "2"}
    rc = subprocess.run(["bash", "/code/pilot/baseten/job.sh"], cwd="/code", env=env).returncode
    stop.set()
    runs.commit()
    return rc


@app.local_entrypoint()
def main():
    download.remote()
    rc = job.remote()
    print(f"[job] exit {rc}")
