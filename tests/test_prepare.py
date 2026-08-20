from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from turn_detector.config import DataConfig
from turn_detector.data.prepare import prepare_dataset
from turn_detector.data.records import read_manifest


def encoded_audio(*, pause: bool = False, sample_rate: int = 16_000) -> bytes:
    time_a = np.arange(round(0.6 * sample_rate)) / sample_rate
    first = 0.2 * np.sin(2 * np.pi * 220 * time_a)
    if pause:
        time_b = np.arange(round(0.7 * sample_rate)) / sample_rate
        second = 0.2 * np.sin(2 * np.pi * 280 * time_b)
        audio = np.concatenate([first, np.zeros(round(0.3 * sample_rate)), second])
    else:
        audio = first
    buffer = BytesIO()
    sf.write(buffer, audio.astype(np.float32), sample_rate, format="FLAC")
    return buffer.getvalue()


def test_prepare_filters_languages_and_builds_causal_examples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        {
            "id": "hin-pause",
            "audio": {"bytes": encoded_audio(pause=True)},
            "language": "hin",
            "endpoint_bool": True,
            "midfiller": True,
            "endfiller": False,
            "synthetic": False,
            "dataset": "fixture",
        },
        {
            "id": "eng-simple",
            "audio": {"bytes": encoded_audio()},
            "language": "eng",
            "endpoint_bool": False,
            "midfiller": False,
            "endfiller": True,
            "synthetic": True,
            "dataset": "fixture",
        },
        {
            "id": "fra-excluded",
            "audio": {"bytes": encoded_audio()},
            "language": "fra",
            "endpoint_bool": True,
            "dataset": "fixture",
        },
    ]
    config = DataConfig(
        output_dir=tmp_path,
        validation_fraction=0.2,
        min_internal_pause_ms=150,
        max_internal_pause_ms=500,
        min_future_speech_ms=500,
    )
    summary = prepare_dataset(config, repo_id="fixture/repo", forced_split="test", rows=rows)
    records = list(read_manifest(tmp_path / "test.jsonl"))
    assert summary["accepted_parent_utterances"] == 2
    assert {record.language for record in records} == {"hin", "eng"}
    causal = [record for record in records if record.example_kind == "causal_internal_pause"]
    assert len(causal) == 1
    assert causal[0].endpoint_bool is False
    parent = next(record for record in records if record.id == "hin-pause")
    assert causal[0].duplicate_group == parent.duplicate_group
    assert all(record.resolved_audio_path(tmp_path / "test.jsonl").exists() for record in records)
    captured = capsys.readouterr()
    progress_output = captured.out + captured.err
    assert "[prepare:test] START" in progress_output
    assert "[prepare:test] COMPLETE" in progress_output
    assert "rows_seen=3" in progress_output
