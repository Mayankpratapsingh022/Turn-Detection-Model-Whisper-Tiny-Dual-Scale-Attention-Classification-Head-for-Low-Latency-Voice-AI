import numpy as np

from turn_detector.config import AppConfig
from turn_detector.data.records import AudioRecord
from turn_detector.evaluation.calibration import apply_temperature, select_policy_from_scored


def record(
    record_id: str,
    parent_id: str,
    *,
    endpoint: bool,
    kind: str = "original",
    pause_ms: int | None = None,
) -> AudioRecord:
    return AudioRecord(
        id=record_id,
        parent_id=parent_id,
        audio_path=f"{record_id}.flac",
        source_repo="fixture/repo",
        language="hin",
        endpoint_bool=endpoint,
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
        duplicate_group=parent_id,
    )


def test_temperature_scaling_and_policy_selection() -> None:
    rows = [
        (record("a", "a", endpoint=True, pause_ms=10_000), 0.9),
        (
            record(
                "a-pause",
                "a",
                endpoint=False,
                kind="causal_internal_pause",
                pause_ms=700,
            ),
            0.2,
        ),
        (record("b", "b", endpoint=False, pause_ms=300), 0.1),
        (record("c", "c", endpoint=True, pause_ms=10_000), 0.8),
        (record("d", "d", endpoint=False, pause_ms=300), 0.25),
    ]
    scored = [
        {
            "id": item.id,
            "parent_id": item.parent_id,
            "label": item.label,
            "probability": probability,
            "record": item,
        }
        for item, probability in rows
    ]
    policy, report = select_policy_from_scored(scored, AppConfig())
    assert policy.temperature > 0
    assert 0 <= policy.threshold <= 1
    assert report["selection_split"] == "validation"
    assert np.allclose(apply_temperature([0.1, 0.9], 1.0), [0.1, 0.9])
