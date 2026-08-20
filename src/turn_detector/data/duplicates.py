from __future__ import annotations

import hashlib

import numpy as np
from scipy import signal

from turn_detector.audio import ensure_float32_mono


def waveform_hash(audio: np.ndarray) -> str:
    waveform = np.clip(ensure_float32_mono(audio), -1.0, 1.0)
    pcm16 = np.round(waveform * 32_767).astype("<i2", copy=False)
    return hashlib.sha256(pcm16.tobytes()).hexdigest()


def acoustic_fingerprint(
    audio: np.ndarray,
    sample_rate: int = 16_000,
    *,
    time_bins: int = 32,
    frequency_bins: int = 24,
) -> str:
    """Return a stable coarse fingerprint for exact and near-duplicate grouping."""

    waveform = ensure_float32_mono(audio)
    if waveform.size == 0:
        return hashlib.sha256(b"empty").hexdigest()
    _, _, spectrum = signal.stft(
        waveform,
        fs=sample_rate,
        nperseg=400,
        noverlap=240,
        nfft=512,
        boundary=None,
        padded=False,
    )
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    if magnitude.size == 0:
        magnitude = np.zeros((257, 1), dtype=np.float32)
    frequency_edges = np.linspace(0, magnitude.shape[0], frequency_bins + 1, dtype=int)
    time_edges = np.linspace(0, magnitude.shape[1], time_bins + 1, dtype=int)
    pooled = np.zeros((frequency_bins, time_bins), dtype=np.float32)
    for f_index in range(frequency_bins):
        f_start, f_end = frequency_edges[f_index : f_index + 2]
        f_end = max(f_start + 1, f_end)
        for t_index in range(time_bins):
            t_start, t_end = time_edges[t_index : t_index + 2]
            t_end = max(t_start + 1, t_end)
            pooled[f_index, t_index] = float(magnitude[f_start:f_end, t_start:t_end].mean())
    pooled -= pooled.mean()
    scale = float(pooled.std()) or 1.0
    quantized = np.clip(np.round((pooled / scale) * 8), -32, 31).astype(np.int8)
    duration_bucket = round(waveform.size / sample_rate, 1)
    digest = hashlib.sha256()
    digest.update(quantized.tobytes())
    digest.update(f"{duration_bucket:.1f}".encode())
    return digest.hexdigest()


def duplicate_group(audio_hash: str, fingerprint: str, parent_group: str | None = None) -> str:
    if parent_group:
        return parent_group
    # Exact duplicates naturally share this fingerprint, while small codec or
    # gain differences can still collapse into the same coarse quantization.
    # Keep audio_hash in the API because it is also persisted for exact-match
    # audits; adding it to this key would defeat near-duplicate grouping.
    del audio_hash
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()
    return digest[:24]
