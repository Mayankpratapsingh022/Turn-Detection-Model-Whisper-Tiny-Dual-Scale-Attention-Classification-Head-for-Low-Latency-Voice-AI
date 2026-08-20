from __future__ import annotations

import json
import os
import platform
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from turn_detector.audio import apply_corruption, load_audio
from turn_detector.config import AppConfig
from turn_detector.data.records import AudioRecord, read_manifest, write_json
from turn_detector.evaluation.metrics import (
    PausePrediction,
    binary_classification_metrics,
    bootstrap_metric,
    bootstrap_policy,
    evaluate_policy,
    operating_point,
    pareto_frontier,
    policy_sweep,
)
from turn_detector.inference import TurnDetector
from turn_detector.progress import log_event, progress_bar


class AudioScorer(Protocol):
    def score(self, audio: np.ndarray, sample_rate: int) -> Any: ...


def _slice_names(record: AudioRecord) -> list[str]:
    names = [
        "all",
        f"language:{record.language}",
        f"source:{record.source_dataset}",
        f"synthetic:{record.synthetic}",
        f"kind:{record.example_kind}",
        f"speech_mix:{record.speech_mix}",
    ]
    if record.midfiller:
        names.append("filler:mid")
    if record.endfiller:
        names.append("filler:end")
    if record.filler_present:
        names.append("filler:any")
    if not record.endpoint_bool and record.filler_present:
        names.append("hard:incomplete_filler")
    if record.duration_seconds < 2:
        names.append("duration:short")
    elif record.duration_seconds > 8:
        names.append("duration:long")
    else:
        names.append("duration:medium")
    return names


def score_manifest(
    detector: AudioScorer,
    manifest_path: str | Path,
    *,
    limit: int | None = None,
    progress_description: str = "score manifest",
) -> list[dict[str, Any]]:
    manifest = Path(manifest_path)
    with manifest.open("r", encoding="utf-8") as handle:
        total = sum(1 for line in handle if line.strip())
    if limit is not None:
        total = min(total, limit)
    scored: list[dict[str, Any]] = []
    record_source = read_manifest(manifest)
    if limit is not None:
        record_source = islice(record_source, limit)
    records = progress_bar(
        record_source,
        total=total,
        description=progress_description,
        unit="clips",
    )
    for record in records:
        waveform, sample_rate = load_audio(record.resolved_audio_path(manifest))
        started = time.perf_counter()
        prediction = detector.score(waveform[-record.valid_samples :], sample_rate)
        end_to_end_ms = (time.perf_counter() - started) * 1_000
        scored.append(
            {
                "id": record.id,
                "parent_id": record.parent_id,
                "label": record.label,
                "probability": prediction.probability,
                "inference_ms": prediction.inference_ms,
                "end_to_end_ms": end_to_end_ms,
                "pause_duration_ms": record.pause_duration_ms,
                "language": record.language,
                "slices": _slice_names(record),
                "record": record,
            }
        )
    return scored


def _metrics_by_slice(scored: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    slices: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        for slice_name in row["slices"]:
            slices[slice_name].append(row)
    results: dict[str, Any] = {}
    for slice_name, rows in sorted(slices.items()):
        labels = [row["label"] for row in rows]
        if len(set(labels)) < 2:
            results[slice_name] = {
                "count": len(rows),
                "positive_rate": float(np.mean(labels)),
                "note": "single-label slice; ranking metrics omitted",
            }
            continue
        results[slice_name] = binary_classification_metrics(
            labels,
            [row["probability"] for row in rows],
            threshold=threshold,
        )
    return results


def _causal_predictions(scored: list[dict[str, Any]]) -> list[PausePrediction]:
    predictions: list[PausePrediction] = []
    for row in scored:
        record: AudioRecord = row["record"]
        if record.example_kind == "causal_internal_pause":
            label = "hold"
            duration = record.pause_duration_ms or 0
        elif record.endpoint_bool:
            label = "eot"
            duration = max(record.pause_duration_ms or 0, 10_000)
        else:
            continue
        predictions.append(
            PausePrediction(
                id=record.id,
                parent_id=record.parent_id,
                label=label,
                probability=float(row["probability"]),
                silence_duration_ms=duration,
                language=record.language,
                slice_name=record.speech_mix,
            )
        )
    return predictions


def _filler_category(record: AudioRecord) -> str:
    if record.midfiller and record.endfiller:
        return "mid_and_end"
    if record.midfiller:
        return "mid"
    if record.endfiller:
        return "end"
    if record.midfiller is None and record.endfiller is None:
        return "unknown"
    return "none"


def _robustness_stratum(record: AudioRecord) -> tuple[str, ...]:
    return (
        record.language,
        f"label:{record.label}",
        f"kind:{record.example_kind}",
        f"filler:{_filler_category(record)}",
        f"synthetic:{record.synthetic}",
    )


def _stratified_robustness_sample(
    records: list[AudioRecord], *, limit: int, seed: int
) -> list[AudioRecord]:
    """Deterministically round-robin across the evaluation-critical intersections."""

    if limit <= 0 or not records:
        return []
    strata: dict[tuple[str, ...], list[AudioRecord]] = defaultdict(list)
    for record in records:
        strata[_robustness_stratum(record)].append(record)

    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, ...], list[AudioRecord]] = {}
    for key, rows in strata.items():
        ordered = sorted(rows, key=lambda row: row.id)
        permutation = rng.permutation(len(ordered))
        buckets[key] = [ordered[int(index)] for index in permutation]

    selected: list[AudioRecord] = []
    positions = {key: 0 for key in buckets}
    active = sorted(buckets)
    while active and len(selected) < min(limit, len(records)):
        remaining: list[tuple[str, ...]] = []
        for key in active:
            position = positions[key]
            if position < len(buckets[key]) and len(selected) < limit:
                selected.append(buckets[key][position])
                positions[key] += 1
            if positions[key] < len(buckets[key]):
                remaining.append(key)
        active = remaining
    return selected


def _robustness_sampling_summary(
    population: list[AudioRecord],
    selected: list[AudioRecord],
    *,
    requested_limit: int,
    seed: int,
) -> dict[str, Any]:
    def counts(rows: list[AudioRecord], field: str) -> dict[str, int]:
        if field == "label":
            values = ("complete" if row.label else "hold" for row in rows)
        elif field == "filler":
            values = (_filler_category(row) for row in rows)
        else:
            values = (str(getattr(row, field)) for row in rows)
        return dict(sorted(Counter(values).items()))

    dimensions = ("language", "label", "example_kind", "filler", "synthetic")
    return {
        "method": "deterministic_equal_round_robin_over_composite_strata",
        "seed": seed,
        "requested_limit": requested_limit,
        "population_count": len(population),
        "selected_count": len(selected),
        "composite_strata_in_population": len({_robustness_stratum(row) for row in population}),
        "composite_strata_selected": len({_robustness_stratum(row) for row in selected}),
        "stratification_dimensions": list(dimensions),
        "population": {field: counts(population, field) for field in dimensions},
        "selected": {field: counts(selected, field) for field in dimensions},
    }


def _robustness_suite(
    detector: TurnDetector,
    manifest_path: str | Path,
    selected: list[AudioRecord],
    *,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    corruption_names = [
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
    results: dict[str, Any] = {}
    with progress_bar(
        total=len(corruption_names) * len(selected),
        description="evaluation robustness",
        unit="scores",
    ) as progress:
        for corruption in corruption_names:
            labels: list[int] = []
            probabilities: list[float] = []
            slice_rows: list[dict[str, Any]] = []
            progress.set_postfix(corruption=corruption, refresh=False)
            for index, record in enumerate(selected):
                waveform, sample_rate = load_audio(record.resolved_audio_path(manifest_path))
                semantic = waveform[-record.valid_samples :]
                corrupted = apply_corruption(
                    semantic,
                    corruption,  # type: ignore[arg-type]
                    sample_rate=sample_rate,
                    seed=seed + index,
                )
                labels.append(record.label)
                probability = detector.score(corrupted, sample_rate).probability
                probabilities.append(probability)
                slice_rows.append(
                    {
                        "label": record.label,
                        "probability": probability,
                        "slices": _slice_names(record),
                    }
                )
                progress.update(1)
            if len(set(labels)) == 2:
                results[corruption] = {
                    **binary_classification_metrics(labels, probabilities, threshold=threshold),
                    "slices": _metrics_by_slice(slice_rows, threshold),
                }
    return results


def evaluate(
    model_path: str | Path,
    config: AppConfig,
    *,
    manifest_path: str | Path | None = None,
    limit: int | None = None,
    run_robustness: bool = True,
) -> dict[str, Any]:
    manifest = Path(manifest_path or config.evaluation.test_manifest)
    log_event(
        "evaluation",
        "START",
        model=model_path,
        manifest=manifest,
        limit=limit or "none",
        robustness=run_robustness,
        bootstrap_samples=config.evaluation.bootstrap_samples,
    )
    detector = TurnDetector(model_path)
    policy = detector.policy
    scored = score_manifest(
        detector,
        manifest,
        limit=limit,
        progress_description="evaluation candidate",
    )
    if not scored:
        raise ValueError("Evaluation manifest is empty")
    labels = [row["label"] for row in scored]
    probabilities = [row["probability"] for row in scored]
    static: dict[str, Any] = binary_classification_metrics(
        labels, probabilities, threshold=policy.threshold
    )
    if len({row["parent_id"] for row in scored}) >= 2 and len(set(labels)) == 2:
        log_event(
            "evaluation:bootstrap",
            "START",
            groups=len({row["parent_id"] for row in scored}),
            samples=config.evaluation.bootstrap_samples,
            metrics=3,
        )
        groups = [row["parent_id"] for row in scored]
        static["group_bootstrap_95ci"] = {
            metric: bootstrap_metric(
                labels,
                probabilities,
                groups,
                metric=metric,
                threshold=policy.threshold,
                samples=config.evaluation.bootstrap_samples,
                seed=config.evaluation.seed,
            )
            for metric in ("f1", "false_cutoff_rate", "auroc")
        }
        log_event("evaluation:bootstrap", "COMPLETE")

    causal = _causal_predictions(scored)
    sweep = (
        policy_sweep(
            causal,
            thresholds=config.evaluation.thresholds,
            action_delays_ms=config.evaluation.action_delays_ms,
            timeouts_ms=config.evaluation.timeouts_ms,
        )
        if causal
        else []
    )
    operating_points = {
        str(target): (
            point.as_dict()
            if (point := operating_point(sweep, max_false_cutoff_rate=target)) is not None
            else None
        )
        for target in config.evaluation.target_false_cutoff_rates
    }
    deployed_policy = (
        evaluate_policy(
            causal,
            threshold=policy.threshold,
            action_delay_ms=policy.min_silence_ms,
            timeout_ms=policy.timeout_ms,
        )
        if causal
        else None
    )
    deployed_bootstrap = (
        bootstrap_policy(
            causal,
            threshold=policy.threshold,
            action_delay_ms=policy.min_silence_ms,
            timeout_ms=policy.timeout_ms,
            samples=config.evaluation.bootstrap_samples,
            seed=config.evaluation.seed,
        )
        if causal
        else None
    )
    records = [row["record"] for row in scored]
    robustness_limit = min(config.evaluation.robustness_limit_per_slice, len(records))
    robustness_records = _stratified_robustness_sample(
        records,
        limit=robustness_limit,
        seed=config.evaluation.seed,
    )
    robustness_sampling = _robustness_sampling_summary(
        records,
        robustness_records,
        requested_limit=robustness_limit,
        seed=config.evaluation.seed,
    )
    robustness = (
        _robustness_suite(
            detector,
            manifest,
            robustness_records,
            threshold=policy.threshold,
            seed=config.evaluation.seed,
        )
        if run_robustness
        else {}
    )
    if run_robustness:
        log_event(
            "evaluation:robustness",
            "COMPLETE",
            corruptions=len(robustness),
            examples=len(robustness_records),
            strata=robustness_sampling["composite_strata_selected"],
        )
    slice_metrics = _metrics_by_slice(scored, policy.threshold)
    report = {
        "model_path": str(model_path),
        "manifest": str(manifest),
        "selection_split": "test",
        "test_policy_tuning_performed": False,
        "static": static,
        "policy": policy.model_dump(mode="json"),
        "slices": slice_metrics,
        "causal": {
            "pause_predictions": len(causal),
            "turns": len({row.parent_id for row in causal}),
            "deployed_policy": deployed_policy.as_dict() if deployed_policy else None,
            "deployed_policy_group_bootstrap_95ci": deployed_bootstrap,
            "operating_points": operating_points,
            "pareto_frontier": [result.as_dict() for result in pareto_frontier(sweep)],
        },
        "robustness": robustness,
        "robustness_sampling": robustness_sampling if run_robustness else None,
        "latency": {
            "count": len(scored),
            "model_p50_ms": float(np.percentile([row["inference_ms"] for row in scored], 50)),
            "model_p95_ms": float(np.percentile([row["inference_ms"] for row in scored], 95)),
            "model_p99_ms": float(np.percentile([row["inference_ms"] for row in scored], 99)),
            "end_to_end_p50_ms": float(np.percentile([row["end_to_end_ms"] for row in scored], 50)),
            "end_to_end_p95_ms": float(np.percentile([row["end_to_end_ms"] for row in scored], 95)),
            "end_to_end_p99_ms": float(np.percentile([row["end_to_end_ms"] for row in scored], 99)),
            "contract": (
                "Per-clip score timing. End-to-end includes waveform standardization, "
                "Whisper log-Mel extraction, calibration, and ONNX inference; it excludes audio "
                "decode/disk I/O and policy silence waiting."
            ),
        },
        "runtime_environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "onnxruntime": getattr(__import__("onnxruntime"), "__version__", "unknown"),
            "intra_op_threads": 1,
        },
    }
    output_dir = config.evaluation.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, output_dir / "evaluation_report.json")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            serializable = {key: value for key, value in row.items() if key != "record"}
            handle.write(json.dumps(serializable))
            handle.write("\n")
    with (output_dir / "policy_sweep.jsonl").open("w", encoding="utf-8") as handle:
        for result in sweep:
            handle.write(json.dumps(result.as_dict()))
            handle.write("\n")
    log_event(
        "evaluation",
        "COMPLETE",
        examples=len(scored),
        causal_predictions=len(causal),
        policy_sweep_points=len(sweep),
        slices=len(slice_metrics),
        output=output_dir / "evaluation_report.json",
    )
    return report


def benchmark_latency(
    detector: TurnDetector,
    audio: np.ndarray,
    sample_rate: int,
    *,
    warmup: int = 20,
    iterations: int = 1_000,
) -> dict[str, float | int]:
    for _ in range(warmup):
        detector.score(audio, sample_rate)
    timings: list[float] = []
    end_to_end: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        prediction = detector.score(audio, sample_rate)
        end_to_end.append((time.perf_counter() - started) * 1_000)
        timings.append(prediction.inference_ms)
    report = {
        "iterations": iterations,
        "model_p50_ms": float(np.percentile(timings, 50)),
        "model_p95_ms": float(np.percentile(timings, 95)),
        "model_p99_ms": float(np.percentile(timings, 99)),
        "end_to_end_p50_ms": float(np.percentile(end_to_end, 50)),
        "end_to_end_p95_ms": float(np.percentile(end_to_end, 95)),
        "end_to_end_p99_ms": float(np.percentile(end_to_end, 99)),
    }
    report["meets_model_p95_100ms_target"] = report["model_p95_ms"] < 100
    return report
