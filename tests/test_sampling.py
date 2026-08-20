import numpy as np

from turn_detector.data.records import AudioRecord
from turn_detector.data.sampling import compute_sampling_weights, enforce_hard_negative_fraction


def record(record_id: str, endpoint: bool, weight: float) -> AudioRecord:
    return AudioRecord(
        id=record_id,
        parent_id=record_id,
        audio_path=f"{record_id}.flac",
        source_repo="fixture/repo",
        language="hin",
        endpoint_bool=endpoint,
        duration_seconds=1.0,
        valid_samples=16_000,
        speech_seconds=0.8,
        speech_ratio=0.8,
        peak_dbfs=-3.0,
        rms_dbfs=-12.0,
        clipping_ratio=0.0,
        silence_ratio=0.2,
        audio_hash=record_id,
        acoustic_fingerprint=record_id,
        duplicate_group=record_id,
        sampling_weight=weight,
    )


def test_sampling_weight_is_not_applied_twice() -> None:
    records = [record("a", False, 1.0), record("b", True, 2.0)]
    weights = compute_sampling_weights(records)
    assert np.isclose(weights[1] / weights[0], 2.0)


def test_hard_negative_sampler_mass_is_explicit() -> None:
    records = [
        record("a", False, 1.0),
        record("b", True, 1.0),
        record("hard", False, 3.0).model_copy(update={"is_hard_negative": True}),
    ]
    weights = enforce_hard_negative_fraction(
        compute_sampling_weights(records), records, fraction=0.30
    )
    hard_mass = weights[-1] / weights.sum()
    assert np.isclose(hard_mass, 0.30)
