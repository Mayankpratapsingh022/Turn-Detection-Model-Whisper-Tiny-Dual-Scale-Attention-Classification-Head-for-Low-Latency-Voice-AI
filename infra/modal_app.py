"""Dormant Modal entrypoints.

Importing this module defines jobs but does not create resources or launch work.
Create the volume explicitly only after cost approval:

    modal volume create hinglish-turn-data
    modal run infra/modal_app.py::prepare
    modal run infra/modal_app.py::train_model
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "hinglish-turn-detector"
VOLUME_NAME = os.environ.get("TURN_DETECTOR_MODAL_VOLUME", "hinglish-turn-data")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "accelerate>=1.2,<2",
        "datasets>=3.2,<5",
        "huggingface-hub>=0.27,<2",
        "matplotlib>=3.9,<4",
        "numpy>=1.26,<3",
        "onnx>=1.17,<2",
        "onnxruntime-gpu>=1.20,<2",
        "pandas>=2.2,<3",
        "pydantic>=2.8,<3",
        "pyarrow>=18,<22",
        "PyYAML>=6.0.2,<7",
        "rich>=13.9,<15",
        "safetensors>=0.5,<1",
        "scikit-learn>=1.5,<2",
        "scipy>=1.13,<2",
        "soundfile>=0.12.1,<1",
        "torch>=2.5,<3",
        "transformers>=4.47,<5",
        "typer>=0.15,<1",
    )
    .add_local_dir("src", remote_path="/workspace/src", copy=True)
    .add_local_dir("configs", remote_path="/workspace/configs", copy=True)
)


def _run(*arguments: str) -> None:
    environment = {**os.environ, "PYTHONPATH": "/workspace/src"}
    subprocess.run(
        ["python", "-m", "turn_detector.cli", *arguments],
        cwd="/workspace",
        env=environment,
        check=True,
    )


def _link_artifacts_to_volume() -> None:
    path = Path("/workspace/artifacts")
    if path.is_symlink():
        return
    if path.exists():
        raise RuntimeError(f"Refusing to replace existing remote path: {path}")
    path.symlink_to("/vol", target_is_directory=True)


@app.function(
    image=image,
    cpu=16,
    memory=65_536,
    timeout=24 * 60 * 60,
    volumes={"/vol": volume},
)
def prepare() -> None:
    _run("data", "prepare", "--config", "configs/modal.yaml")
    volume.commit()


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=16,
    memory=65_536,
    timeout=24 * 60 * 60,
    volumes={"/vol": volume},
)
def train_model() -> None:
    _run("train", "--config", "configs/modal.yaml")
    volume.commit()


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=16,
    memory=65_536,
    timeout=24 * 60 * 60,
    volumes={"/vol": volume},
)
def train_experiment(experiment: str = "e5_causal_filler") -> None:
    allowed = {
        "e2_global",
        "e3_focused",
        "e4_dual",
        "e5_causal_filler",
        "e6_hard_negative",
    }
    if experiment not in allowed:
        raise ValueError(f"Unknown experiment {experiment!r}; choose one of {sorted(allowed)}")
    _link_artifacts_to_volume()
    _run("train", "--config", f"configs/experiments/{experiment}.yaml")
    volume.commit()


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=16,
    memory=65_536,
    timeout=12 * 60 * 60,
    volumes={"/vol": volume},
)
def mine_hard_negatives() -> None:
    _run(
        "mine-hard-negatives",
        "--model-path",
        "/vol/checkpoints/e5_causal_filler/best",
        "--manifest",
        "/vol/data/train.jsonl",
        "--output",
        "/vol/data/hard_negatives.jsonl",
        "--config",
        "configs/modal.yaml",
    )
    volume.commit()


@app.function(
    image=image,
    cpu=16,
    memory=65_536,
    timeout=24 * 60 * 60,
    volumes={"/vol": volume},
)
def evaluate_and_export(experiment: str = "main") -> None:
    if experiment == "main":
        checkpoint = "/vol/checkpoints/best"
        export_dir = "/vol/export/main"
    elif experiment in {
        "e2_global",
        "e3_focused",
        "e4_dual",
        "e5_causal_filler",
        "e6_hard_negative",
    }:
        checkpoint = f"/vol/checkpoints/{experiment}/best"
        export_dir = f"/vol/export/{experiment}"
    else:
        raise ValueError(f"Unknown experiment: {experiment!r}")
    model_path = f"{export_dir}/hinglish-turn.int8.onnx"
    _run(
        "export",
        "--checkpoint",
        checkpoint,
        "--output",
        f"{export_dir}/hinglish-turn.onnx",
        "--config",
        "configs/modal.yaml",
    )
    _run(
        "calibrate",
        "--model-path",
        model_path,
        "--config",
        "configs/modal.yaml",
    )
    _run(
        "evaluate",
        "--model-path",
        model_path,
        "--config",
        "configs/modal.yaml",
        "--output-dir",
        f"/vol/evaluation/{experiment}",
    )
    volume.commit()
