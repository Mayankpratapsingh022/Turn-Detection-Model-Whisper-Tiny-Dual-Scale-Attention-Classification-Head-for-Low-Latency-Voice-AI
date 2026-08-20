from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from turn_detector.data import hinglish
from turn_detector.data.hinglish import classify_speech_mix
from turn_detector.data.records import AudioRecord, read_manifest, write_manifest


def test_mixed_script_hinglish() -> None:
    assert (
        classify_speech_mix("मुझे cab booking cancel करनी है", "hin", asr_confidence=0.9)
        == "hinglish_high_confidence"
    )


def test_roman_hinglish() -> None:
    assert (
        classify_speech_mix("mujhe kal office leave chahiye", "eng", asr_confidence=0.9)
        == "hinglish_high_confidence"
    )


def test_low_confidence_is_uncertain() -> None:
    assert classify_speech_mix("mujhe cab chahiye", "hin", asr_confidence=0.1) == "uncertain"


def test_plain_languages() -> None:
    assert classify_speech_mix("मुझे कल घर जाना है", "hin", asr_confidence=0.9) == "hindi"
    assert classify_speech_mix("please cancel my booking", "eng", asr_confidence=0.9) == "english"


def test_asr_tagging_checkpoints_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeModel:
        def transcribe(self, *_args: object, **_kwargs: object) -> tuple[list[object], None]:
            return [SimpleNamespace(text="mujhe office leave chahiye", avg_logprob=-0.1)], None

    monkeypatch.setattr(hinglish, "_load_asr_model", lambda *_args: FakeModel())
    audio = 0.2 * np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000)
    records = []
    for index in range(2):
        sf.write(tmp_path / f"{index}.flac", audio.astype(np.float32), 16_000)
        records.append(
            AudioRecord(
                id=str(index),
                parent_id=str(index),
                audio_path=f"{index}.flac",
                source_repo="fixture/repo",
                language="hin",
                endpoint_bool=bool(index),
                duration_seconds=1.0,
                valid_samples=16_000,
                speech_seconds=1.0,
                speech_ratio=1.0,
                peak_dbfs=-3.0,
                rms_dbfs=-12.0,
                clipping_ratio=0.0,
                silence_ratio=0.0,
                audio_hash=str(index),
                acoustic_fingerprint=str(index),
                duplicate_group=str(index),
            )
        )
    source = tmp_path / "train.jsonl"
    output = tmp_path / "train.tagged.jsonl"
    write_manifest(records, source)
    first = hinglish.tag_manifest_with_asr(source, output, limit=1, checkpoint_every=1)
    assert first["processed"] == 1
    assert len(list(read_manifest(output))) == 2
    second = hinglish.tag_manifest_with_asr(source, output, checkpoint_every=1)
    assert second["processed"] == 1
    assert second["skipped_existing"] == 1
    assert all(record.speech_mix != "untagged" for record in read_manifest(output))
