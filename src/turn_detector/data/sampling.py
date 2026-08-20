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


def focused_sampling_weights(
    records: list[AudioRecord],
    *,
    hindi_fraction: float,
    hard_negative_fraction: float,
) -> np.ndarray:
    """Build an explicit language/endpoint sampler while preserving within-cell priorities.

    Base examples receive the requested Hindi/English mass and balanced COMPLETE/HOLD mass inside
    each language. Hard negatives receive their own global mass; that mass is distributed across
    whichever languages contain mined examples, and the base allocation compensates so the final
    language target remains exact.
    """

    if not records:
        return np.zeros(0, dtype=np.float64)
    if not 0 < hindi_fraction < 1:
        raise ValueError("hindi_fraction must be between 0 and 1")
    if not 0 <= hard_negative_fraction < 1:
        raise ValueError("hard_negative_fraction must be in [0, 1)")

    initial = np.asarray([record.sampling_weight for record in records], dtype=np.float64)
    if not np.isfinite(initial).all() or np.any(initial <= 0):
        raise ValueError("Sampling weights must be finite and positive")
    languages = np.asarray([record.language for record in records])
    endpoints = np.asarray([record.endpoint_bool for record in records], dtype=bool)
    hard = np.asarray([record.is_hard_negative for record in records], dtype=bool)
    base = ~hard
    desired = {"hin": hindi_fraction, "eng": 1.0 - hindi_fraction}
    available_languages = [language for language in ("hin", "eng") if np.any(languages == language)]
    desired_total = sum(desired[language] for language in available_languages)
    language_targets = {
        language: desired[language] / desired_total for language in available_languages
    }

    result = np.zeros_like(initial)
    target_hard = hard_negative_fraction if hard.any() and base.any() else float(hard.all())
    hard_languages = [
        language for language in available_languages if np.any(hard & (languages == language))
    ]
    hard_language_targets: dict[str, float] = {language: 0.0 for language in available_languages}
    if target_hard and hard_languages:
        hard_target_denominator = sum(language_targets[language] for language in hard_languages)
        for language in hard_languages:
            mass = target_hard * language_targets[language] / hard_target_denominator
            if mass > language_targets[language] + 1e-12:
                raise ValueError(
                    "Hard-negative target is incompatible with the requested language mass"
                )
            hard_language_targets[language] = mass
            mask = hard & (languages == language)
            result[mask] = initial[mask] * (mass / initial[mask].sum())

    for language in available_languages:
        language_base_mass = language_targets[language] - hard_language_targets[language]
        language_base = base & (languages == language)
        present_labels = [
            label for label in (False, True) if np.any(language_base & (endpoints == label))
        ]
        if not present_labels:
            if language_base_mass > 1e-12:
                raise ValueError(f"No base examples available for language {language!r}")
            continue
        label_mass = language_base_mass / len(present_labels)
        for label in present_labels:
            mask = language_base & (endpoints == label)
            result[mask] = initial[mask] * (label_mass / initial[mask].sum())

    if not np.isclose(result.sum(), 1.0, atol=1e-8):
        raise RuntimeError(f"Focused sampling mass sums to {result.sum():.8f}, expected 1.0")
    return result / result.mean()


def sampling_mass_summary(weights: np.ndarray, records: list[AudioRecord]) -> dict[str, float]:
    """Report the probability mass a weighted sampler assigns to important training slices."""

    values = np.asarray(weights, dtype=np.float64)
    if len(values) != len(records) or values.sum() <= 0:
        raise ValueError("Weights must align with a non-empty record list")
    probabilities = values / values.sum()

    def mass(predicate: object) -> float:
        return float(probabilities[np.asarray(predicate, dtype=bool)].sum())

    return {
        "hindi_fraction": mass([record.language == "hin" for record in records]),
        "english_fraction": mass([record.language == "eng" for record in records]),
        "complete_fraction": mass([record.endpoint_bool for record in records]),
        "hold_fraction": mass([not record.endpoint_bool for record in records]),
        "filler_fraction": mass([bool(record.filler_present) for record in records]),
        "causal_pause_fraction": mass(
            [record.example_kind == "causal_internal_pause" for record in records]
        ),
        "hard_negative_fraction": mass([record.is_hard_negative for record in records]),
    }
