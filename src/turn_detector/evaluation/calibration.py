from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from turn_detector.config import AppConfig, PolicyConfig
from turn_detector.data.records import AudioRecord, write_json
from turn_detector.evaluation.evaluator import _causal_predictions, score_manifest
from turn_detector.evaluation.metrics import (
    binary_classification_metrics,
    fit_temperature,
    operating_point,
    policy_sweep,
    tpr_at_fpr,
)
from turn_detector.inference import TurnDetector
from turn_detector.progress import log_event


def probability_to_logit(probability: float) -> float:
    clipped = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return math.log(clipped / (1 - clipped))


def apply_temperature(probabilities: list[float], temperature: float) -> list[float]:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    logits = np.asarray([probability_to_logit(value) for value in probabilities]) / temperature
    return (1 / (1 + np.exp(-logits))).tolist()


def select_policy_from_scored(
    scored: list[dict[str, Any]],
    config: AppConfig,
    *,
    target_false_cutoff_rate: float = 0.05,
) -> tuple[PolicyConfig, dict[str, Any]]:
    """Fit temperature on original validation clips and tune policy on causal pauses."""

    originals = [
        row
        for row in scored
        if isinstance(row.get("record"), AudioRecord) and row["record"].example_kind == "original"
    ]
    labels = [int(row["label"]) for row in originals]
    if len(set(labels)) < 2:
        raise ValueError("Calibration needs both COMPLETE and HOLD validation examples")

    # score_manifest returns probabilities after the current policy temperature.
    # Recover the raw model logit so repeated calibration is idempotent.
    raw_logits = [
        probability_to_logit(float(row["probability"])) * config.policy.temperature
        for row in originals
    ]
    temperature = fit_temperature(labels, raw_logits)
    calibrated_originals = apply_temperature(
        [float(row["probability"]) for row in originals],
        temperature / config.policy.temperature,
    )

    calibrated_rows: list[dict[str, Any]] = []
    calibrated_all = apply_temperature(
        [float(row["probability"]) for row in scored],
        temperature / config.policy.temperature,
    )
    for row, probability in zip(scored, calibrated_all, strict=True):
        calibrated_rows.append({**row, "probability": probability})

    causal = _causal_predictions(calibrated_rows)
    selected_point = None
    sweep = []
    if causal:
        # A denser threshold grid is worthwhile here because this is the only
        # split where threshold selection is permitted.
        thresholds = [float(round(float(value), 2)) for value in np.linspace(0.05, 0.95, 91)]
        sweep = policy_sweep(
            causal,
            thresholds=thresholds,
            action_delays_ms=config.evaluation.action_delays_ms,
            timeouts_ms=config.evaluation.timeouts_ms,
        )
        selected_point = operating_point(sweep, max_false_cutoff_rate=target_false_cutoff_rate)
        if selected_point is None:
            selected_point = min(
                sweep,
                key=lambda row: (row.false_cutoff_rate, row.mean_endpoint_latency_ms),
            )

    if selected_point is None:
        _, threshold = tpr_at_fpr(
            labels,
            calibrated_originals,
            target_fpr=target_false_cutoff_rate,
        )
        min_silence_ms = config.policy.min_silence_ms
        timeout_ms = config.policy.timeout_ms
    else:
        threshold = selected_point.threshold
        min_silence_ms = selected_point.action_delay_ms
        timeout_ms = selected_point.timeout_ms

    policy = config.policy.model_copy(
        update={
            "temperature": temperature,
            "threshold": threshold,
            "min_silence_ms": min_silence_ms,
            "timeout_ms": timeout_ms,
        }
    )
    before = binary_classification_metrics(
        labels,
        [float(row["probability"]) for row in originals],
        threshold=config.policy.threshold,
    )
    after = binary_classification_metrics(
        labels,
        calibrated_originals,
        threshold=policy.threshold,
    )
    report = {
        "examples": len(originals),
        "causal_pause_predictions": len(causal),
        "target_false_cutoff_rate": target_false_cutoff_rate,
        "policy": policy.model_dump(mode="json"),
        "before": before,
        "after": after,
        "selected_causal_operating_point": (
            selected_point.as_dict() if selected_point is not None else None
        ),
        "selection_split": "validation",
        "test_split_used": False,
    }
    return policy, report


def calibrate(
    model_path: str | Path,
    config: AppConfig,
    *,
    manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
    target_false_cutoff_rate: float = 0.05,
    limit: int | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path or config.train.validation_manifest)
    log_event(
        "calibration",
        "START",
        model=model_path,
        manifest=manifest,
        target_false_cutoff_rate=target_false_cutoff_rate,
        limit=limit or "none",
    )
    detector = TurnDetector(model_path, policy=config.policy)
    scored = score_manifest(
        detector,
        manifest,
        limit=limit,
        progress_description="calibration validation",
    )
    policy, report = select_policy_from_scored(
        scored,
        config,
        target_false_cutoff_rate=target_false_cutoff_rate,
    )
    destination = (
        Path(output_path) if output_path is not None else Path(model_path).parent / "policy.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(policy.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        **report,
        "model_path": str(model_path),
        "manifest": str(manifest),
        "policy_path": str(destination),
    }
    write_json(report, destination.with_name("calibration_report.json"))
    log_event(
        "calibration",
        "COMPLETE",
        examples=report["examples"],
        causal_predictions=report["causal_pause_predictions"],
        temperature=f"{policy.temperature:.5f}",
        threshold=f"{policy.threshold:.5f}",
        min_silence_ms=policy.min_silence_ms,
        timeout_ms=policy.timeout_ms,
        output=destination,
    )
    return report
