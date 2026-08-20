import numpy as np

from turn_detector.audio import (
    analyze_quality,
    apply_corruption,
    find_internal_pauses,
    resample_audio,
    standardize_candidate_audio,
)


def tone(seconds: float, sample_rate: int = 16_000, frequency: float = 220) -> np.ndarray:
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    return (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_standardization_is_left_padded_and_has_fixed_trailing_silence() -> None:
    sample_rate = 16_000
    audio = np.concatenate([tone(0.6), np.zeros(round(0.7 * sample_rate), dtype=np.float32)])
    standardized = standardize_candidate_audio(
        audio, sample_rate, max_seconds=2, trailing_silence_ms=200
    )
    assert standardized.waveform.shape == (32_000,)
    assert np.allclose(standardized.waveform[-3_000:], 0)
    assert 0 < standardized.valid_samples < standardized.waveform.size
    assert standardized.mel_frame_mask(n_frames=200).sum() == np.ceil(
        standardized.valid_samples / 160
    )


def test_internal_pause_requires_future_speech() -> None:
    sample_rate = 16_000
    audio = np.concatenate(
        [tone(0.5), np.zeros(round(0.3 * sample_rate)), tone(0.7, frequency=280)]
    )
    pauses = find_internal_pauses(
        audio,
        sample_rate,
        min_pause_ms=150,
        max_pause_ms=500,
        min_future_speech_ms=500,
    )
    assert len(pauses) == 1
    assert 200 <= pauses[0].duration_ms <= 350


def test_quality_and_resampling() -> None:
    audio = tone(1.0)
    quality = analyze_quality(audio)
    assert quality.valid
    assert quality.speech_seconds > 0.8
    downsampled = resample_audio(audio, 16_000, 8_000)
    assert abs(len(downsampled) - 8_000) <= 1


def test_quality_rejects_non_finite_audio() -> None:
    audio = tone(1.0)
    audio[100] = np.nan
    quality = analyze_quality(audio)
    assert not quality.valid
    assert quality.reason == "non_finite"


def test_corruptions_are_finite() -> None:
    audio = tone(0.5)
    for name in ("telephone", "mulaw", "noise_5db", "gain_low", "clipping", "reverb"):
        corrupted = apply_corruption(audio, name)  # type: ignore[arg-type]
        assert corrupted.ndim == 1
        assert np.isfinite(corrupted).all()
