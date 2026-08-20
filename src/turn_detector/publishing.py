from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _first_existing(paths: tuple[Path, ...], description: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    choices = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Missing {description}; checked: {choices}")


def stage_model_release(
    checkpoint_dir: Path,
    export_dir: Path,
    output_dir: Path,
    *,
    evaluation_dir: Path | None = None,
    model_card: Path | None = None,
) -> dict[str, Any]:
    """Assemble a self-contained Hub folder without optimizer state or source audio."""
    repository_root = Path(__file__).resolve().parents[2]
    checkpoint_config = _first_existing(
        (checkpoint_dir / "turn_detector_config.json",), "checkpoint model config"
    )
    checkpoint_weights = _first_existing(
        (checkpoint_dir / "model.safetensors", checkpoint_dir / "pytorch_model.bin"),
        "checkpoint model weights",
    )
    int8_model = _first_existing(tuple(sorted(export_dir.glob("*.int8.onnx"))), "INT8 ONNX model")
    export_report = _first_existing((export_dir / "export_report.json",), "export report")
    policy = _first_existing((export_dir / "policy.json",), "calibrated/default policy")
    selected_card = model_card or repository_root / "MODEL_CARD_TEMPLATE.md"
    selected_license = repository_root / "LICENSE"
    for required in (selected_card, selected_license):
        if not required.exists():
            raise FileNotFoundError(required)

    sources: list[tuple[Path, Path]] = [
        (checkpoint_config, Path("turn_detector_config.json")),
        (checkpoint_weights, Path(checkpoint_weights.name)),
        (int8_model, Path(int8_model.name)),
        (export_report, Path("export_report.json")),
        (policy, Path("policy.json")),
        (selected_card, Path("README.md")),
        (selected_license, Path("LICENSE")),
    ]
    fp32_models = sorted(export_dir.glob("*.onnx"))
    for model in fp32_models:
        if model != int8_model:
            sources.append((model, Path(model.name)))
    optional_reports = [
        export_dir / "calibration_report.json",
        checkpoint_dir.parent / "training_report.json",
        checkpoint_dir.parent / "resolved_config.yaml",
    ]
    for report in optional_reports:
        if report.exists():
            sources.append((report, Path(report.name)))
    if evaluation_dir is not None and evaluation_dir.exists():
        allowed_suffixes = {".json", ".csv", ".png", ".md"}
        for source in sorted(evaluation_dir.rglob("*")):
            if source.is_file() and source.suffix.lower() in allowed_suffixes:
                relative = source.relative_to(evaluation_dir)
                sources.append((source, Path("evaluation") / relative))

    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    seen_destinations: set[Path] = set()
    for source, relative in sources:
        if relative in seen_destinations:
            continue
        seen_destinations.add(relative)
        destination = output_dir / relative
        _copy(source, destination)
        copied.append(
            {
                "path": relative.as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )

    manifest = {
        "format_version": 1,
        "checkpoint_source": str(checkpoint_dir),
        "export_source": str(export_dir),
        "files": copied,
        "excluded": ["trainer_state.pt", "source_audio", ".env"],
    }
    manifest_path = output_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"release_dir": str(output_dir), **manifest}


def push_model_to_hub(
    folder: Path,
    repo_id: str,
    *,
    private: bool = True,
    revision: str = "main",
    create_pr: bool = False,
    commit_message: str = "Upload trained Hinglish turn detector",
    acknowledge_source_license_review: bool = False,
) -> dict[str, Any]:
    """Upload a staged release. This is never called automatically by training."""
    if not acknowledge_source_license_review:
        raise ValueError(
            "Publishing requires --acknowledge-source-license-review because the upstream "
            "dataset card does not declare an explicit license."
        )
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Add it to .env before publishing.")
    manifest_path = folder / "release_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Package the model first; release_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allow_patterns = [entry["path"] for entry in manifest["files"]]
    allow_patterns.append("release_manifest.json")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Hub publishing requires `uv sync --extra hub`") from exc
    api = HfApi(token=token)
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        create_pr=create_pr,
        commit_message=commit_message,
        allow_patterns=allow_patterns,
        token=token,
    )
    return {
        "repo_id": repo_id,
        "repo_url": str(repo_url),
        "commit_url": getattr(commit, "commit_url", None),
        "revision": revision,
        "private": private,
        "uploaded_files": allow_patterns,
    }
