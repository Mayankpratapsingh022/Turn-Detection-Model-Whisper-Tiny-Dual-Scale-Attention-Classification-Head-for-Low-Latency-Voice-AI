from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from turn_detector.config import AppConfig
from turn_detector.hub_cache import cache_huggingface_assets, pin_config_to_cached_revisions
from turn_detector.publishing import push_model_to_hub, stage_model_release


def test_cache_assets_records_resolved_revisions_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path / "models--repo" / "snapshots" / f"sha-{len(calls)}")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-secret")
    destination = tmp_path / "cache_manifest.json"
    result = cache_huggingface_assets(AppConfig(), manifest_path=destination)

    assert len(result["assets"]) == 3
    assert result["assets"][0]["resolved_revision"] == "sha-1"
    assert calls[0]["repo_type"] == "dataset"
    assert "dummy-hf-secret" not in destination.read_text()

    pinned_path = tmp_path / "pinned.yaml"
    pin_config_to_cached_revisions(AppConfig(), destination, pinned_path)
    pinned = AppConfig.from_yaml(pinned_path)
    assert pinned.data.train_revision == "sha-1"
    assert pinned.data.test_revision == "sha-2"
    assert pinned.model.base_model_revision == "sha-3"


def _make_release_inputs(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoints" / "best"
    export_dir = tmp_path / "export"
    checkpoint.mkdir(parents=True)
    export_dir.mkdir()
    (checkpoint / "turn_detector_config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint.parent / "training_report.json").write_text("{}")
    (export_dir / "hinglish-turn.int8.onnx").write_bytes(b"int8")
    (export_dir / "hinglish-turn.onnx").write_bytes(b"fp32")
    (export_dir / "export_report.json").write_text("{}")
    (export_dir / "policy.json").write_text("{}")
    return checkpoint, export_dir


def test_stage_model_release_contains_runtime_and_training_artifacts(tmp_path: Path) -> None:
    checkpoint, export_dir = _make_release_inputs(tmp_path)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "evaluation_report.json").write_text("{}")
    output = tmp_path / "release"

    result = stage_model_release(checkpoint, export_dir, output, evaluation_dir=evaluation)
    staged = {entry["path"] for entry in result["files"]}
    assert "model.safetensors" in staged
    assert "hinglish-turn.int8.onnx" in staged
    assert "evaluation/evaluation_report.json" in staged
    assert "trainer_state.pt" in result["excluded"]
    assert (output / "release_manifest.json").exists()


def test_push_model_is_private_and_uploads_only_manifest_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, export_dir = _make_release_inputs(tmp_path)
    folder = tmp_path / "release"
    stage_model_release(checkpoint, export_dir, folder)
    calls: dict[str, dict[str, Any]] = {}

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "dummy-token"

        def create_repo(self, **kwargs: Any) -> str:
            calls["create"] = kwargs
            return "https://huggingface.invalid/owner/model"

        def upload_folder(self, **kwargs: Any) -> Any:
            calls["upload"] = kwargs
            return SimpleNamespace(commit_url="https://huggingface.invalid/commit")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    result = push_model_to_hub(
        folder,
        "owner/model",
        acknowledge_source_license_review=True,
    )

    assert calls["create"]["private"] is True
    assert ".env" not in calls["upload"]["allow_patterns"]
    assert "release_manifest.json" in calls["upload"]["allow_patterns"]
    assert result["private"] is True


def test_push_requires_explicit_license_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="acknowledge-source-license-review"):
        push_model_to_hub(tmp_path, "owner/model")
