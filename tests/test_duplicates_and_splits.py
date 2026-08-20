from pathlib import Path

import numpy as np
import pytest

from turn_detector.data.duplicates import acoustic_fingerprint, duplicate_group, waveform_hash
from turn_detector.data.prepare import remove_cross_corpus_duplicates
from turn_detector.data.records import AudioRecord, read_manifest, write_manifest
from turn_detector.data.splits import assert_no_group_leakage, assign_grouped_stratified_splits


def record(index: int, group: str) -> AudioRecord:
    return AudioRecord(
        id=f"id-{index}",
        parent_id=f"id-{index}",
        audio_path=f"audio/{index}.flac",
        source_repo="test/repo",
        source_dataset="source",
        language="hin",
        endpoint_bool=bool(index % 2),
        duration_seconds=1.0,
        valid_samples=16_000,
        speech_seconds=0.8,
        speech_ratio=0.8,
        peak_dbfs=-3.0,
        rms_dbfs=-12.0,
        clipping_ratio=0.0,
        silence_ratio=0.2,
        audio_hash=f"hash-{index}",
        acoustic_fingerprint=f"fp-{index}",
        duplicate_group=group,
    )


def test_hashes_are_stable_and_signal_sensitive() -> None:
    audio = np.linspace(-0.2, 0.2, 16_000, dtype=np.float32)
    assert waveform_hash(audio) == waveform_hash(audio.copy())
    assert acoustic_fingerprint(audio) == acoustic_fingerprint(audio.copy())
    changed = audio.copy()
    changed[100:500] *= -1
    assert waveform_hash(audio) != waveform_hash(changed)


def test_near_duplicates_group_by_acoustic_fingerprint() -> None:
    assert duplicate_group("exact-a", "coarse-fp") == duplicate_group("exact-b", "coarse-fp")


def test_grouped_split_has_no_leakage() -> None:
    records = [record(index, f"group-{index // 2}") for index in range(40)]
    assigned = assign_grouped_stratified_splits(records, validation_fraction=0.2)
    assert {item.split for item in assigned} == {"train", "validation"}
    assert_no_group_leakage(assigned)


def test_cross_corpus_dedup_removes_entire_parent() -> None:
    reference = [record(0, "shared")]
    candidate = [
        record(1, "shared"),
        record(2, "shared").model_copy(update={"parent_id": "id-1"}),
        record(3, "unique"),
    ]
    filtered, report = remove_cross_corpus_duplicates(reference, candidate)
    assert [item.id for item in filtered] == ["id-3"]
    assert report["removed_records"] == 2


def test_leakage_check_fails() -> None:
    records = [
        record(0, "same").model_copy(update={"split": "train"}),
        record(1, "same").model_copy(update={"split": "validation"}),
    ]
    with pytest.raises(ValueError, match="crosses"):
        assert_no_group_leakage(records)


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    expected = [record(0, "g0"), record(1, "g1")]
    assert write_manifest(expected, path) == 2
    assert list(read_manifest(path)) == expected
