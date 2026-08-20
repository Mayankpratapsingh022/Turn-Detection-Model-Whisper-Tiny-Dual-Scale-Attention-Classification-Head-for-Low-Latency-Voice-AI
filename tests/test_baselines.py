from pathlib import Path
from typing import Any

import pytest

from turn_detector.config import AppConfig, PolicyConfig
from turn_detector.data.records import AudioRecord
from turn_detector.evaluation import baselines


def _record(
    record_id: str,
    *,
    endpoint: bool,
    kind: str = "original",
    pause_ms: int | None = None,
) -> AudioRecord:
    return AudioRecord(
        id=record_id,
        parent_id=record_id.split("-")[0],
        audio_path=f"{record_id}.flac",
        source_repo="fixture/repo",
        language="hin",
        endpoint_bool=endpoint,
        midfiller=kind == "causal_internal_pause",
        endfiller=False,
        synthetic=False,
        example_kind=kind,
        duration_seconds=1.0,
        valid_samples=16_000,
        speech_seconds=0.8,
        speech_ratio=0.8,
        peak_dbfs=-3.0,
        rms_dbfs=-12.0,
        clipping_ratio=0.0,
        silence_ratio=0.2,
        pause_duration_ms=pause_ms,
        audio_hash=record_id,
        acoustic_fingerprint=record_id,
        duplicate_group=record_id.split("-")[0],
    )


def _scored(rows: list[tuple[AudioRecord, float]]) -> list[dict[str, Any]]:
    return [
        {
            "id": record.id,
            "parent_id": record.parent_id,
            "label": record.label,
            "probability": probability,
            "inference_ms": 4.0,
            "end_to_end_ms": 9.0,
            "pause_duration_ms": record.pause_duration_ms,
            "language": record.language,
            "slices": ["all", f"language:{record.language}"],
            "record": record,
        }
        for record, probability in rows
    ]


def test_public_baseline_is_calibrated_on_validation_then_frozen_for_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_path = tmp_path / "candidate.onnx"
    public_path = tmp_path / "public.onnx"
    candidate_path.write_bytes(b"candidate")
    public_path.write_bytes(b"public")
    validation_manifest = tmp_path / "validation.jsonl"
    test_manifest = tmp_path / "test.jsonl"
    validation_manifest.write_text("fixture")
    test_manifest.write_text("fixture")

    validation_rows = _scored(
        [
            (_record("a-eot", endpoint=True, pause_ms=10_000), 0.85),
            (_record("b-hold", endpoint=False, pause_ms=300), 0.20),
            (
                _record(
                    "a-pause",
                    endpoint=False,
                    kind="causal_internal_pause",
                    pause_ms=700,
                ),
                0.15,
            ),
            (_record("c-eot", endpoint=True, pause_ms=10_000), 0.75),
            (_record("d-hold", endpoint=False, pause_ms=300), 0.25),
        ]
    )
    candidate_test = _scored(
        [
            (_record("x-eot", endpoint=True, pause_ms=10_000), 0.90),
            (_record("y-hold", endpoint=False, pause_ms=300), 0.10),
            (
                _record(
                    "x-pause",
                    endpoint=False,
                    kind="causal_internal_pause",
                    pause_ms=700,
                ),
                0.10,
            ),
            (_record("z-eot", endpoint=True, pause_ms=10_000), 0.80),
        ]
    )
    public_test = _scored(
        [
            (_record("x-eot", endpoint=True, pause_ms=10_000), 0.80),
            (_record("y-hold", endpoint=False, pause_ms=300), 0.30),
            (
                _record(
                    "x-pause",
                    endpoint=False,
                    kind="causal_internal_pause",
                    pause_ms=700,
                ),
                0.25,
            ),
            (_record("z-eot", endpoint=True, pause_ms=10_000), 0.70),
        ]
    )

    class FakeCandidate:
        def __init__(self, _path: str | Path) -> None:
            self.policy = PolicyConfig(
                threshold=0.6,
                temperature=1.0,
                min_silence_ms=200,
                timeout_ms=1_000,
            )

    class FakePublic:
        def __init__(self, _path: str | Path | None = None) -> None:
            self.model_path = public_path

    def fake_score_manifest(
        detector: object, manifest: str | Path, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        if Path(manifest) == validation_manifest:
            return validation_rows
        return candidate_test if isinstance(detector, FakeCandidate) else public_test

    monkeypatch.setattr(baselines, "TurnDetector", FakeCandidate)
    monkeypatch.setattr(baselines, "SmartTurnV32Baseline", FakePublic)
    monkeypatch.setattr(baselines, "score_manifest", fake_score_manifest)
    config = AppConfig().model_copy(
        update={
            "train": AppConfig().train.model_copy(
                update={"validation_manifest": validation_manifest}
            ),
            "evaluation": AppConfig().evaluation.model_copy(
                update={
                    "test_manifest": test_manifest,
                    "output_dir": tmp_path / "evaluation",
                    "bootstrap_samples": 20,
                }
            ),
        }
    )

    report = baselines.compare_baselines(candidate_path, config)

    assert report["baseline_policy_selection_split"] == "validation"
    assert report["test_policy_tuning_performed"] is False
    assert report["smart_turn_v3.2"]["validation_calibration"]["selection_split"] == "validation"
    assert report["smart_turn_v3.2"]["policy"]["temperature"] > 0
    assert report["smart_turn_v3.2"]["model_latency"]["end_to_end_p95_ms"] == 9.0
    assert (
        tmp_path / "evaluation" / "baselines" / "smart_turn_v3.2_validation_predictions.jsonl"
    ).exists()
