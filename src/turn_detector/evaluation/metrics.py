from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy import optimize, stats


def _as_arrays(
    labels: list[int] | np.ndarray, probabilities: list[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.ndim != 1 or y_true.shape != y_prob.shape:
        raise ValueError("labels and probabilities must be equal-length 1D arrays")
    if y_true.size == 0:
        raise ValueError("At least one prediction is required")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    return y_true, y_prob


def roc_auc_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = stats.rankdata(probabilities, method="average")
    positive_rank_sum = float(ranks[positives].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def average_precision_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive_count = int(labels.sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    cumulative_positives = np.cumsum(sorted_labels)
    precision = cumulative_positives / np.arange(1, labels.size + 1)
    return float((precision * sorted_labels).sum() / positive_count)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        selected = indices == index
        if not selected.any():
            continue
        confidence = float(probabilities[selected].mean())
        accuracy = float(labels[selected].mean())
        error += float(selected.mean()) * abs(confidence - accuracy)
    return error


def binary_classification_metrics(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y_true, y_prob = _as_arrays(labels, probabilities)
    predicted = y_prob >= threshold
    positive = y_true == 1
    tp = int((predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": int(y_true.size),
        "threshold": threshold,
        "accuracy": (tp + tn) / y_true.size,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_cutoff_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_hold_rate": fn / (fn + tp) if fn + tp else 0.0,
        "auroc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "brier": float(np.mean(np.square(y_prob - y_true))),
        "ece": expected_calibration_error(y_true, y_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def tpr_at_fpr(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    *,
    target_fpr: float = 0.05,
) -> tuple[float, float]:
    y_true, y_prob = _as_arrays(labels, probabilities)
    best_tpr = 0.0
    best_threshold = 1.0
    for threshold in np.unique(np.concatenate([y_prob, np.asarray([0.0, 1.0])])):
        prediction = y_prob >= threshold
        negatives = y_true == 0
        positives = y_true == 1
        fpr = float((prediction & negatives).sum() / max(1, negatives.sum()))
        tpr = float((prediction & positives).sum() / max(1, positives.sum()))
        if fpr <= target_fpr and tpr >= best_tpr:
            best_tpr, best_threshold = tpr, float(threshold)
    return best_tpr, best_threshold


def fit_temperature(labels: list[int], logits: list[float]) -> float:
    y_true = np.asarray(labels, dtype=np.float64)
    raw_logits = np.asarray(logits, dtype=np.float64)

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        scaled = raw_logits / temperature
        # Stable binary cross entropy with logits.
        return float(
            np.mean(np.maximum(scaled, 0) - scaled * y_true + np.log1p(np.exp(-np.abs(scaled))))
        )

    result = optimize.minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    return float(math.exp(result.x))


@dataclass(frozen=True, slots=True)
class PausePrediction:
    id: str
    parent_id: str
    label: str  # "hold" or "eot"
    probability: float
    silence_duration_ms: int
    language: str = "unknown"
    slice_name: str = "all"


@dataclass(frozen=True, slots=True)
class PolicyResult:
    threshold: float
    action_delay_ms: int
    timeout_ms: int
    false_cutoff_rate: float
    mean_endpoint_latency_ms: float
    median_endpoint_latency_ms: float
    p95_endpoint_latency_ms: float
    turns: int
    false_cutoff_turns: int
    endpoint_turns: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_policy(
    predictions: list[PausePrediction],
    *,
    threshold: float,
    action_delay_ms: int,
    timeout_ms: int,
) -> PolicyResult:
    grouped: dict[str, list[PausePrediction]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction.parent_id, []).append(prediction)
    false_cutoff_turns = 0
    endpoint_latencies: list[int] = []
    for rows in grouped.values():
        cut = False
        for row in rows:
            decision_delay = action_delay_ms if row.probability >= threshold else timeout_ms
            if row.label == "hold" and decision_delay < row.silence_duration_ms:
                cut = True
            elif row.label == "eot":
                endpoint_latencies.append(decision_delay)
        false_cutoff_turns += int(cut)
    latency_array = np.asarray(endpoint_latencies, dtype=np.float64)
    if latency_array.size:
        mean_latency = float(latency_array.mean())
        median_latency = float(np.median(latency_array))
        p95_latency = float(np.percentile(latency_array, 95))
    else:
        mean_latency = median_latency = p95_latency = float("nan")
    return PolicyResult(
        threshold=threshold,
        action_delay_ms=action_delay_ms,
        timeout_ms=timeout_ms,
        false_cutoff_rate=false_cutoff_turns / max(1, len(grouped)),
        mean_endpoint_latency_ms=mean_latency,
        median_endpoint_latency_ms=median_latency,
        p95_endpoint_latency_ms=p95_latency,
        turns=len(grouped),
        false_cutoff_turns=false_cutoff_turns,
        endpoint_turns=len(endpoint_latencies),
    )


def policy_sweep(
    predictions: list[PausePrediction],
    *,
    thresholds: list[float] | tuple[float, ...],
    action_delays_ms: list[int] | tuple[int, ...],
    timeouts_ms: list[int] | tuple[int, ...],
) -> list[PolicyResult]:
    return [
        evaluate_policy(
            predictions,
            threshold=threshold,
            action_delay_ms=action_delay,
            timeout_ms=timeout,
        )
        for threshold in thresholds
        for action_delay in action_delays_ms
        for timeout in timeouts_ms
        if timeout >= action_delay
    ]


def pareto_frontier(results: list[PolicyResult]) -> list[PolicyResult]:
    ordered = sorted(
        results,
        key=lambda result: (result.false_cutoff_rate, result.mean_endpoint_latency_ms),
    )
    frontier: list[PolicyResult] = []
    best_latency = float("inf")
    for result in ordered:
        if result.mean_endpoint_latency_ms < best_latency:
            frontier.append(result)
            best_latency = result.mean_endpoint_latency_ms
    return frontier


def operating_point(
    results: list[PolicyResult], *, max_false_cutoff_rate: float
) -> PolicyResult | None:
    eligible = [result for result in results if result.false_cutoff_rate <= max_false_cutoff_rate]
    return min(eligible, key=lambda result: result.mean_endpoint_latency_ms, default=None)


def bootstrap_metric(
    labels: list[int],
    probabilities: list[float],
    group_ids: list[str],
    *,
    metric: str = "f1",
    threshold: float = 0.5,
    samples: int = 2_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    if not (len(labels) == len(probabilities) == len(group_ids)):
        raise ValueError("labels, probabilities and group_ids must align")
    unique_groups = np.unique(np.asarray(group_ids))
    group_to_indices = {
        group: np.flatnonzero(np.asarray(group_ids) == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        selected_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([group_to_indices[group] for group in selected_groups])
        result = binary_classification_metrics(
            np.asarray(labels)[indices],
            np.asarray(probabilities)[indices],
            threshold=threshold,
        )
        value = float(result[metric])
        if math.isfinite(value):
            values.append(value)
    point = float(binary_classification_metrics(labels, probabilities, threshold=threshold)[metric])
    if not values:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_policy(
    predictions: list[PausePrediction],
    *,
    threshold: float,
    action_delay_ms: int,
    timeout_ms: int,
    samples: int = 2_000,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Turn-group bootstrap intervals for the deployed policy."""

    grouped: dict[str, list[PausePrediction]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction.parent_id, []).append(prediction)
    group_ids = sorted(grouped)
    if not group_ids:
        raise ValueError("At least one causal turn is required")
    rng = np.random.default_rng(seed)
    false_cutoff_rates: list[float] = []
    endpoint_latencies: list[float] = []
    for _ in range(samples):
        selected = rng.choice(group_ids, size=len(group_ids), replace=True)
        sampled: list[PausePrediction] = []
        for copy_index, group_id in enumerate(selected):
            sampled.extend(
                replace(row, parent_id=f"{copy_index}:{group_id}") for row in grouped[str(group_id)]
            )
        result = evaluate_policy(
            sampled,
            threshold=threshold,
            action_delay_ms=action_delay_ms,
            timeout_ms=timeout_ms,
        )
        false_cutoff_rates.append(result.false_cutoff_rate)
        if math.isfinite(result.mean_endpoint_latency_ms):
            endpoint_latencies.append(result.mean_endpoint_latency_ms)
    point = evaluate_policy(
        predictions,
        threshold=threshold,
        action_delay_ms=action_delay_ms,
        timeout_ms=timeout_ms,
    )

    def interval(value: float, bootstraps: list[float]) -> tuple[float, float, float]:
        if not bootstraps:
            return value, float("nan"), float("nan")
        return value, float(np.percentile(bootstraps, 2.5)), float(np.percentile(bootstraps, 97.5))

    return {
        "false_cutoff_rate": interval(point.false_cutoff_rate, false_cutoff_rates),
        "mean_endpoint_latency_ms": interval(point.mean_endpoint_latency_ms, endpoint_latencies),
    }


def paired_group_bootstrap_delta(
    labels: list[int],
    probabilities_a: list[float],
    probabilities_b: list[float],
    group_ids: list[str],
    *,
    metric: str = "f1",
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    samples: int = 2_000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired group bootstrap for candidate-minus-baseline metric deltas."""

    if not (len(labels) == len(probabilities_a) == len(probabilities_b) == len(group_ids)):
        raise ValueError("All paired inputs must align")
    y_true = np.asarray(labels)
    a = np.asarray(probabilities_a)
    b = np.asarray(probabilities_b)
    groups = np.asarray(group_ids)
    unique_groups = np.unique(groups)
    group_to_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)

    def delta(indices: np.ndarray) -> float:
        metric_a = float(
            binary_classification_metrics(y_true[indices], a[indices], threshold=threshold_a)[
                metric
            ]
        )
        metric_b = float(
            binary_classification_metrics(y_true[indices], b[indices], threshold=threshold_b)[
                metric
            ]
        )
        return metric_a - metric_b

    values: list[float] = []
    for _ in range(samples):
        selected_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([group_to_indices[group] for group in selected_groups])
        value = delta(indices)
        if math.isfinite(value):
            values.append(value)
    point = delta(np.arange(y_true.size))
    return {
        "delta": point,
        "ci_low": float(np.percentile(values, 2.5)) if values else float("nan"),
        "ci_high": float(np.percentile(values, 97.5)) if values else float("nan"),
        "probability_candidate_better": float(np.mean(np.asarray(values) > 0))
        if values
        else float("nan"),
    }


def mcnemar_test(
    labels: list[int],
    probabilities_a: list[float],
    probabilities_b: list[float],
    *,
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
) -> dict[str, float | int]:
    y_true = np.asarray(labels, dtype=np.int64)
    correct_a = (np.asarray(probabilities_a) >= threshold_a) == y_true
    correct_b = (np.asarray(probabilities_b) >= threshold_b) == y_true
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p_value": p_value}
