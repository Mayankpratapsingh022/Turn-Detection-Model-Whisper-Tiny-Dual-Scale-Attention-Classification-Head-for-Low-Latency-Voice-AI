from __future__ import annotations

from collections import Counter

import numpy as np

from turn_detector.data.records import AudioRecord


def category_weight(record: AudioRecord) -> float:
    if record.speech_mix == "hinglish_high_confidence":
        weight = 4.0
    elif record.language == "hin" and record.filler_present:
        weight = 2.5
    elif record.language == "eng" and record.filler_present:
        weight = 2.0
    elif record.language == "hin":
        weight = 1.6
    else:
        weight = 1.0
    if record.synthetic is False:
        weight *= 2.0
    if record.is_hard_negative:
        weight *= 3.0
    if record.example_kind == "causal_internal_pause":
        weight *= 1.5
    return weight


def compute_sampling_weights(records: list[AudioRecord]) -> np.ndarray:
    if not records:
        return np.zeros(0, dtype=np.float64)
    label_counts = Counter(record.endpoint_bool for record in records)
    total = len(records)
    weights = []
    for record in records:
        balance = total / (2 * label_counts[record.endpoint_bool])
        weights.append(record.sampling_weight * balance)
    result = np.asarray(weights, dtype=np.float64)
    return result / result.mean()


def enforce_hard_negative_fraction(
    weights: np.ndarray,
    records: list[AudioRecord],
    fraction: float,
) -> np.ndarray:
    """Normalize sampler mass so mined negatives occupy an explicit fraction."""

    if not 0 <= fraction < 1:
        raise ValueError("hard-negative fraction must be in [0, 1)")
    result = np.asarray(weights, dtype=np.float64).copy()
    hard = np.asarray([record.is_hard_negative for record in records], dtype=bool)
    if not hard.any() or hard.all():
        return result
    if fraction == 0:
        result[hard] = 0
        return result / result.mean()
    result[hard] *= fraction / result[hard].sum()
    result[~hard] *= (1 - fraction) / result[~hard].sum()
    return result / result.mean()
