"""Baseten training job for pilot phase 2. Push with pilot/baseten/push.sh (stages code + tasks)."""

from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, CheckpointingConfig, Compute, Image, Runtime, TrainingJob, TrainingProject

project = TrainingProject(
    name="samgo-pilot",
    job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
        compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=2), cpu_count=16, memory="200Gi"),
        runtime=Runtime(
            start_commands=["timeout 170m bash job.sh"],
            cache_config=CacheConfig(enabled=True),
            checkpointing_config=CheckpointingConfig(enabled=True),
        ),
    ),
)
