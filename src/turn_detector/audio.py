from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from scipy import signal

FloatArray = np.ndarray


@dataclass(frozen=True, slots=True)
class AudioQuality:
    duration_seconds: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    speech_seconds: float
    valid: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class StandardizedAudio:
    waveform: FloatArray
    valid_samples: int
    speech_end_sample: int
    original_duration_seconds: float

    def mel_frame_mask(self, *, n_frames: int = 800, hop_length: int = 160) -> np.ndarray:
        valid_frames = min(n_frames, math.ceil(self.valid_samples / hop_length))
        mask = np.zeros(n_frames, dtype=np.int64)
        if valid_frames:
            mask[-valid_frames:] = 1
        return mask


@dataclass(frozen=True, slots=True)
class PauseSpan:
    start_sample: int
    end_sample: int
    duration_ms: int
    future_speech_ms: int


def ensure_float32_mono(audio: np.ndarray) -> FloatArray:
    value = np.asarray(audio)
    if value.ndim == 2:
        value = value.mean(axis=1)
    if value.ndim != 1:
        raise ValueError(f"Expected mono or channel-last audio, got shape {value.shape}")
    if np.issubdtype(value.dtype, np.integer):
        scale = max(abs(np.iinfo(value.dtype).min), np.iinfo(value.dtype).max)
        value = value.astype(np.float32) / float(scale)
    else:
        value = value.astype(np.float32, copy=False)
    return np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)


def resample_audio(audio: FloatArray, source_rate: int, target_rate: int = 16_000) -> FloatArray:
    audio = ensure_float32_mono(audio)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    result = signal.resample_poly(audio, target_rate // divisor, source_rate // divisor)
    return result.astype(np.float32, copy=False)


def load_audio(source: str | Path | bytes | bytearray | dict[str, Any]) -> tuple[FloatArray, int]:
    if isinstance(source, dict):
        if source.get("array") is not None:
            return ensure_float32_mono(np.asarray(source["array"])), int(source["sampling_rate"])
        if source.get("bytes") is not None:
            source = source["bytes"]
        elif source.get("path"):
            source = source["path"]
        else:
            raise ValueError("Audio mapping must contain array, bytes, or path")

    if isinstance(source, (bytes, bytearray)):
        audio, rate = sf.read(BytesIO(source), dtype="float32", always_2d=False)
    else:
        audio, rate = sf.read(str(source), dtype="float32", always_2d=False)
    return ensure_float32_mono(audio), int(rate)


def save_audio(path: str | Path, audio: FloatArray, sample_rate: int = 16_000) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, ensure_float32_mono(audio), sample_rate, format="FLAC", subtype="PCM_16")


def _frame_rms(audio: FloatArray, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if audio.size < frame_length:
        padded = np.pad(audio, (0, frame_length - audio.size))
        return np.asarray([np.sqrt(np.mean(np.square(padded), dtype=np.float64))], dtype=np.float32)
    starts = np.arange(0, audio.size - frame_length + 1, hop_length)
    values = np.empty(starts.size, dtype=np.float32)
    for index, start in enumerate(starts):
        frame = audio[start : start + frame_length]
        values[index] = np.sqrt(np.mean(np.square(frame), dtype=np.float64))
    return values


def _smooth_boolean_mask(mask: np.ndarray, min_true: int, max_false_gap: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if result.size == 0:
        return result

    # Fill brief holes between speech regions.
    index = 0
    while index < result.size:
        if result[index]:
            index += 1
            continue
        end = index
        while end < result.size and not result[end]:
            end += 1
        if index > 0 and end < result.size and end - index <= max_false_gap:
            result[index:end] = True
        index = end

    # Remove tiny isolated speech bursts.
    index = 0
    while index < result.size:
        if not result[index]:
            index += 1
            continue
        end = index
        while end < result.size and result[end]:
            end += 1
        if end - index < min_true:
            result[index:end] = False
        index = end
    return result


def speech_frame_mask(
    audio: FloatArray,
    sample_rate: int = 16_000,
    *,
    frame_ms: int = 25,
    hop_ms: int = 10,
    absolute_rms_floor: float = 0.006,
) -> tuple[np.ndarray, int, int]:
    """Return a deterministic energy-VAD mask used for preparation and tests.

    Production streaming can use Silero. This fallback deliberately has no model
    download, which keeps data audits and CI deterministic.
    """

    waveform = ensure_float32_mono(audio)
    frame_length = max(1, round(sample_rate * frame_ms / 1_000))
    hop_length = max(1, round(sample_rate * hop_ms / 1_000))
    rms = _frame_rms(waveform, frame_length, hop_length)
    if rms.size == 0:
        return np.zeros(0, dtype=bool), frame_length, hop_length
    noise_floor = float(np.percentile(rms, 20))
    high_percentile = float(np.percentile(rms, 95))
    # When nearly the entire clip is voiced, the low percentile is speech—not
    # a background-noise estimate. Applying a 2.5x multiplier in that case
    # pushes the threshold above stationary vowels and tones.
    if high_percentile > noise_floor * 1.8:
        threshold = max(absolute_rms_floor, noise_floor * 2.5, high_percentile * 0.08)
    else:
        threshold = max(absolute_rms_floor, high_percentile * 0.08)
    raw = rms >= threshold
    min_true = max(1, round(80 / hop_ms))
    max_gap = max(1, round(60 / hop_ms))
    return _smooth_boolean_mask(raw, min_true, max_gap), frame_length, hop_length


def speech_segments(audio: FloatArray, sample_rate: int = 16_000) -> list[tuple[int, int]]:
    mask, frame_length, hop_length = speech_frame_mask(audio, sample_rate)
    segments: list[tuple[int, int]] = []
    index = 0
    while index < mask.size:
        if not mask[index]:
            index += 1
            continue
        end = index
        while end < mask.size and mask[end]:
            end += 1
        start_sample = index * hop_length
        end_sample = min(len(audio), (end - 1) * hop_length + frame_length)
        segments.append((start_sample, end_sample))
        index = end
    return segments


def find_internal_pauses(
    audio: FloatArray,
    sample_rate: int = 16_000,
    *,
    min_pause_ms: int = 150,
    max_pause_ms: int = 1_200,
    min_future_speech_ms: int = 500,
) -> list[PauseSpan]:
    segments = speech_segments(audio, sample_rate)
    pauses: list[PauseSpan] = []
    for index, (_, current_end) in enumerate(segments[:-1]):
        next_start, _ = segments[index + 1]
        pause_samples = next_start - current_end
        pause_ms = round(pause_samples * 1_000 / sample_rate)
        future_speech_samples = sum(end - start for start, end in segments[index + 1 :])
        future_speech_ms = round(future_speech_samples * 1_000 / sample_rate)
        if min_pause_ms <= pause_ms <= max_pause_ms and future_speech_ms >= min_future_speech_ms:
            pauses.append(
                PauseSpan(
                    start_sample=current_end,
                    end_sample=next_start,
                    duration_ms=pause_ms,
                    future_speech_ms=future_speech_ms,
                )
            )
    return pauses


def standardize_candidate_audio(
    audio: FloatArray,
    sample_rate: int = 16_000,
    *,
    max_seconds: float = 8.0,
    trailing_silence_ms: int = 200,
) -> StandardizedAudio:
    waveform = np.clip(ensure_float32_mono(audio), -1.0, 1.0)
    original_duration = waveform.size / sample_rate
    segments = speech_segments(waveform, sample_rate)
    speech_end = segments[-1][1] if segments else 0
    trailing_samples = round(sample_rate * trailing_silence_ms / 1_000)
    semantic_audio = waveform[:speech_end] if speech_end else np.zeros(0, dtype=np.float32)
    standardized = np.concatenate([semantic_audio, np.zeros(trailing_samples, dtype=np.float32)])
    max_samples = round(max_seconds * sample_rate)
    if standardized.size > max_samples:
        standardized = standardized[-max_samples:]
    valid_samples = standardized.size
    if standardized.size < max_samples:
        standardized = np.pad(standardized, (max_samples - standardized.size, 0))
    return StandardizedAudio(
        waveform=standardized.astype(np.float32, copy=False),
        valid_samples=valid_samples,
        speech_end_sample=speech_end,
        original_duration_seconds=original_duration,
    )


def analyze_quality(audio: FloatArray, sample_rate: int = 16_000) -> AudioQuality:
    raw = np.asarray(audio)
    contains_non_finite = not np.isfinite(raw).all()
    waveform = ensure_float32_mono(audio)
    if waveform.size == 0:
        return AudioQuality(0.0, -120.0, -120.0, 0.0, 1.0, 0.0, False, "empty")
    if contains_non_finite:
        reason = "non_finite"
    elif waveform.size / sample_rate < 0.20:
        reason = "too_short"
    else:
        reason = None
    peak = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    peak_dbfs = 20 * math.log10(max(peak, 1e-6))
    rms_dbfs = 20 * math.log10(max(rms, 1e-6))
    clipping_ratio = float(np.mean(np.abs(waveform) >= 0.999))
    mask, _, hop = speech_frame_mask(waveform, sample_rate)
    speech_seconds = float(mask.sum() * hop / sample_rate)
    silence_ratio = 1.0 - float(mask.mean()) if mask.size else 1.0
    if reason is None and speech_seconds < 0.10:
        reason = "no_speech"
    if reason is None and clipping_ratio > 0.10:
        reason = "severe_clipping"
    return AudioQuality(
        duration_seconds=waveform.size / sample_rate,
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        speech_seconds=speech_seconds,
        valid=reason is None,
        reason=reason,
    )


def add_white_noise(audio: FloatArray, snr_db: float, rng: np.random.Generator) -> FloatArray:
    waveform = ensure_float32_mono(audio)
    signal_power = float(np.mean(np.square(waveform), dtype=np.float64))
    if signal_power <= 1e-12:
        return waveform.copy()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, math.sqrt(noise_power), waveform.shape).astype(np.float32)
    return np.clip(waveform + noise, -1.0, 1.0)


def telephone_bandpass(audio: FloatArray, sample_rate: int = 16_000) -> FloatArray:
    waveform = ensure_float32_mono(audio)
    low = 300 / (sample_rate / 2)
    high = min(0.999, 3_400 / (sample_rate / 2))
    sos = signal.butter(6, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfilt(sos, waveform)
    at_8k = resample_audio(filtered, sample_rate, 8_000)
    return resample_audio(at_8k, 8_000, sample_rate)


def mu_law_roundtrip(audio: FloatArray, mu: int = 255) -> FloatArray:
    waveform = np.clip(ensure_float32_mono(audio), -1.0, 1.0)
    encoded = np.sign(waveform) * np.log1p(mu * np.abs(waveform)) / np.log1p(mu)
    quantized = np.round((encoded + 1.0) * 0.5 * mu) / mu
    restored = 2.0 * quantized - 1.0
    decoded = np.sign(restored) * (np.expm1(np.abs(restored) * np.log1p(mu)) / mu)
    return decoded.astype(np.float32)


CorruptionName = Literal[
    "clean",
    "telephone",
    "mulaw",
    "noise_5db",
    "noise_10db",
    "noise_20db",
    "speed_0.9",
    "speed_1.1",
    "gain_low",
    "clipping",
    "reverb",
]


def apply_corruption(
    audio: FloatArray,
    name: CorruptionName,
    *,
    sample_rate: int = 16_000,
    seed: int = 42,
) -> FloatArray:
    waveform = ensure_float32_mono(audio)
    rng = np.random.default_rng(seed)
    if name == "clean":
        return waveform.copy()
    if name == "telephone":
        return telephone_bandpass(waveform, sample_rate)
    if name == "mulaw":
        return mu_law_roundtrip(waveform)
    if name.startswith("noise_"):
        snr = float(name.removeprefix("noise_").removesuffix("db"))
        return add_white_noise(waveform, snr, rng)
    if name.startswith("speed_"):
        factor = float(name.removeprefix("speed_"))
        target_length = max(1, round(waveform.size / factor))
        return signal.resample(waveform, target_length).astype(np.float32)
    if name == "gain_low":
        return (waveform * 0.2).astype(np.float32)
    if name == "clipping":
        return np.clip(waveform * 3.0, -0.7, 0.7).astype(np.float32)
    if name == "reverb":
        impulse_length = round(0.35 * sample_rate)
        impulse = np.zeros(impulse_length, dtype=np.float32)
        impulse[0] = 1.0
        for delay_ms, gain in ((35, 0.45), (73, 0.28), (127, 0.16), (211, 0.09)):
            impulse[min(impulse_length - 1, round(delay_ms * sample_rate / 1_000))] = gain
        tail = rng.normal(0.0, 1.0, impulse_length).astype(np.float32)
        tail *= np.exp(-np.linspace(0, 7, impulse_length)).astype(np.float32) * 0.025
        impulse += tail
        reverberant = signal.fftconvolve(waveform, impulse, mode="full")[: waveform.size]
        peak = float(np.max(np.abs(reverberant)))
        if peak > 1.0:
            reverberant /= peak
        return reverberant.astype(np.float32)
    raise ValueError(f"Unknown corruption: {name}")
