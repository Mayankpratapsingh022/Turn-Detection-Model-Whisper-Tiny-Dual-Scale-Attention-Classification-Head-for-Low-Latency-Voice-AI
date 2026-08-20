from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from turn_detector.config import AppConfig


class ExperimentTracker(Protocol):
    def log(self, metrics: dict[str, Any], *, step: int, commit: bool = True) -> None: ...

    def set_summary(self, metrics: dict[str, Any]) -> None: ...

    def log_model_artifact(self, checkpoint_dir: Path) -> None: ...

    def finish(self, *, exit_code: int = 0) -> None: ...


class NullTracker:
    def log(self, metrics: dict[str, Any], *, step: int, commit: bool = True) -> None:
        del metrics, step, commit

    def set_summary(self, metrics: dict[str, Any]) -> None:
        del metrics

    def log_model_artifact(self, checkpoint_dir: Path) -> None:
        del checkpoint_dir

    def finish(self, *, exit_code: int = 0) -> None:
        del exit_code


def flatten_metrics(payload: dict[str, Any], *, prefix: str = "") -> dict[str, int | float]:
    """Flatten numeric W&B metrics while dropping text and null values."""
    flattened: dict[str, int | float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, prefix=name))
        elif isinstance(value, bool):
            flattened[name] = int(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            flattened[name] = value
    return flattened


class WandbTracker:
    def __init__(self, config: AppConfig) -> None:
        # python-dotenv preserves NAME= as an empty string, but W&B validates some
        # optional settings (especially WANDB_RUN_ID) before init and rejects empties.
        for variable in ("WANDB_RUN_ID", "WANDB_RUN_NAME", "WANDB_ENTITY"):
            if os.getenv(variable) == "":
                os.environ.pop(variable, None)
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "W&B tracking is enabled; install it with `uv sync --extra tracking`"
            ) from exc

        mode_value = os.getenv("WANDB_MODE", config.tracking.mode)
        if mode_value not in {"online", "offline", "disabled"}:
            raise ValueError("WANDB_MODE must be online, offline, or disabled")
        mode = cast(Literal["online", "offline", "disabled"], mode_value)
        if mode == "online" and not os.getenv("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY is missing. Add it to .env, or set WANDB_MODE=offline."
            )

        project = os.getenv("WANDB_PROJECT", config.tracking.project)
        entity = os.getenv("WANDB_ENTITY") or config.tracking.entity
        run_id = os.getenv("WANDB_RUN_ID") or config.tracking.run_id
        run_name = (
            os.getenv("WANDB_RUN_NAME") or config.tracking.run_name or config.train.output_dir.name
        )
        self._wandb = wandb
        self._log_best_model_artifact = config.tracking.log_best_model_artifact
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            id=run_id,
            resume="allow" if run_id else None,
            mode=mode,
            tags=list(config.tracking.tags),
            config=config.model_dump(mode="json"),
            dir=str(config.train.output_dir),
            job_type="train",
        )
        if self._run is None:
            raise RuntimeError("wandb.init returned no run")
        run_metadata = {
            "id": self._run.id,
            "name": self._run.name,
            "project": project,
            "entity": entity,
            "url": self._run.url,
            "mode": mode,
        }
        metadata_path = config.train.output_dir / "wandb_run.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    def log(self, metrics: dict[str, Any], *, step: int, commit: bool = True) -> None:
        self._run.log(flatten_metrics(metrics), step=step, commit=commit)

    def set_summary(self, metrics: dict[str, Any]) -> None:
        for key, value in flatten_metrics(metrics).items():
            self._run.summary[key] = value

    def log_model_artifact(self, checkpoint_dir: Path) -> None:
        if not self._log_best_model_artifact:
            return
        artifact = self._wandb.Artifact(
            name=f"{self._run.id}-best-checkpoint",
            type="model",
            metadata={"format": "safetensors", "role": "best-validation-checkpoint"},
        )
        for filename in ("model.safetensors", "pytorch_model.bin", "turn_detector_config.json"):
            source = checkpoint_dir / filename
            if source.exists():
                artifact.add_file(str(source), name=filename)
        self._run.log_artifact(artifact, aliases=["best"])

    def finish(self, *, exit_code: int = 0) -> None:
        self._run.finish(exit_code=exit_code)


def initialize_tracker(config: AppConfig) -> ExperimentTracker:
    enabled = config.tracking.enabled
    env_enabled = os.getenv("WANDB_ENABLED")
    if env_enabled is not None:
        normalized = env_enabled.strip().lower()
        if normalized not in {"1", "true", "yes", "0", "false", "no"}:
            raise ValueError("WANDB_ENABLED must be true or false")
        enabled = normalized in {"1", "true", "yes"}
    if not enabled:
        return NullTracker()
    return WandbTracker(config)
