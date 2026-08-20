from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from turn_detector.config import AppConfig
from turn_detector.tracking import NullTracker, WandbTracker, flatten_metrics, initialize_tracker


class FakeRun:
    def __init__(self) -> None:
        self.id = "run-123"
        self.name = "e5"
        self.url = "https://wandb.invalid/run-123"
        self.summary: dict[str, Any] = {}
        self.logged: list[tuple[dict[str, Any], int, bool]] = []
        self.artifacts: list[Any] = []
        self.exit_code: int | None = None

    def log(self, payload: dict[str, Any], *, step: int, commit: bool) -> None:
        self.logged.append((payload, step, commit))

    def log_artifact(self, artifact: Any, *, aliases: list[str]) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code


class FakeArtifact:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.files: list[tuple[str, str]] = []

    def add_file(self, source: str, *, name: str) -> None:
        self.files.append((source, name))


def test_flatten_metrics_keeps_only_finite_numeric_values() -> None:
    assert flatten_metrics(
        {"loss": 1.25, "nested": {"fcr": 0.05}, "bad": float("nan"), "label": "x"}
    ) == {"loss": 1.25, "nested/fcr": 0.05}


def test_wandb_tracker_logs_without_serializing_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_run = FakeRun()
    captured_init: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> FakeRun:
        captured_init.update(kwargs)
        return fake_run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=fake_init, Artifact=lambda **kwargs: FakeArtifact(**kwargs)),
    )
    monkeypatch.setenv("WANDB_API_KEY", "dummy-secret-that-must-not-be-logged")
    monkeypatch.setenv("WANDB_RUN_ID", "")
    config = AppConfig().model_copy(
        update={
            "train": AppConfig().train.model_copy(update={"output_dir": tmp_path}),
            "tracking": AppConfig().tracking.model_copy(update={"enabled": True}),
        }
    )
    tracker = WandbTracker(config)
    tracker.log({"train": {"loss": 0.5}}, step=2)
    tracker.set_summary({"best": 0.9})
    tracker.finish()

    assert fake_run.logged == [({"train/loss": 0.5}, 2, True)]
    assert fake_run.summary == {"best": 0.9}
    assert "dummy-secret" not in str(captured_init)
    assert "dummy-secret" not in (tmp_path / "wandb_run.json").read_text()
    assert "WANDB_RUN_ID" not in os.environ
    assert fake_run.exit_code == 0


def test_wandb_enabled_environment_overrides_disabled_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_ENABLED", "true")
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    config = AppConfig().model_copy(
        update={"train": AppConfig().train.model_copy(update={"output_dir": tmp_path})}
    )
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        initialize_tracker(config)


def test_wandb_disabled_environment_returns_null_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_ENABLED", "false")
    config = AppConfig().model_copy(
        update={"tracking": AppConfig().tracking.model_copy(update={"enabled": True})}
    )
    assert isinstance(initialize_tracker(config), NullTracker)
