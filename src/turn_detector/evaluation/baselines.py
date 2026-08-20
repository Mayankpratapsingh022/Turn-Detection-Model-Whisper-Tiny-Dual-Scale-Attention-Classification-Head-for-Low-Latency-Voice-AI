from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from turn_detector.audio import ensure_float32_mono, resample_audio
from turn_detector.config import AppConfig
from turn_detector.data.records import write_json
from turn_detector.evaluation.calibration import apply_temperature, select_policy_from_scored
from turn_detector.evaluation.evaluator import (
    _causal_predictions,
    _metrics_by_slice,
    score_manifest,
)
from turn_detector.evaluation.metrics import (
    binary_classification_metrics,
    evaluate_policy,
    mcnemar_test,
    paired_group_bootstrap_delta,
)
from turn_detector.inference import TurnDetector
from turn_detector.progress import log_event

SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_FILENAME = "smart-turn-v3.2-cpu.onnx"
# Pin the public v3.2 release so a future repository update cannot silently
# change an experiment that claims to use this baseline.
SMART_TURN_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    probability: float
    inference_ms: float


class SmartTurnV32Baseline:
    """Official Smart Turn v3.2 CPU ONNX preprocessing and inference contract."""

    def __init__(self, model_path: str | Path | None = None, *, intra_op_threads: int = 1) -> None:
        try:
            import onnxruntime as ort
            from transformers import WhisperFeatureExtractor
        except ImportError as exc:  # pragma: no cover - optional dependencies
            raise RuntimeError("Published baseline requires `uv sync --extra baselines`") from exc
        if model_path is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Published baseline requires `uv sync --extra baselines`"
                ) from exc
            model_path = hf_hub_download(
                repo_id=SMART_TURN_REPO,
                filename=SMART_TURN_FILENAME,
                revision=SMART_TURN_REVISION,
            )
        self.model_path = Path(model_path)
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)

    @staticmethod
    def _last_eight_seconds(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        waveform = np.clip(ensure_float32_mono(audio), -1.0, 1.0)
        if sample_rate != 16_000:
            waveform = resample_audio(waveform, sample_rate, 16_000)
        max_samples = 8 * 16_000
        if waveform.size > max_samples:
            return waveform[-max_samples:]
        if waveform.size < max_samples:
            return np.pad(waveform, (max_samples - waveform.size, 0))
        return waveform

    def score(self, audio: np.ndarray, sample_rate: int = 16_000) -> BaselinePrediction:
        waveform = self._last_eight_seconds(audio, sample_rate)
        features = self.feature_extractor(
            waveform,
            sampling_rate=16_000,
            return_tensors="np",
            padding="max_length",
            max_length=8 * 16_000,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        started = time.perf_counter()
        probability = float(
            np.asarray(self.session.run(None, {"input_features": features})[0]).reshape(-1)[0]
        )
        inference_ms = (time.perf_counter() - started) * 1_000
        return BaselinePrediction(probability=probability, inference_ms=inference_ms)


def _probabilities(scored: list[dict[str, Any]]) -> list[float]:
    return [float(row["probability"]) for row in scored]


def _latency(scored: list[dict[str, Any]]) -> dict[str, float]:
    model_values = [float(row["inference_ms"]) for row in scored]
    end_to_end_values = [float(row["end_to_end_ms"]) for row in scored]
    return {
        "model_p50_ms": float(np.percentile(model_values, 50)),
        "model_p95_ms": float(np.percentile(model_values, 95)),
        "model_p99_ms": float(np.percentile(model_values, 99)),
        "end_to_end_p50_ms": float(np.percentile(end_to_end_values, 50)),
        "end_to_end_p95_ms": float(np.percentile(end_to_end_values, 95)),
        "end_to_end_p99_ms": float(np.percentile(end_to_end_values, 99)),
    }


def _temperature_scale_scored(
    scored: list[dict[str, Any]], temperature: float
) -> list[dict[str, Any]]:
    calibrated = apply_temperature(_probabilities(scored), temperature)
    return [
        {**row, "probability": probability}
        for row, probability in zip(scored, calibrated, strict=True)
    ]


def compare_baselines(
    candidate_model_path: str | Path,
    config: AppConfig,
    *,
    manifest_path: str | Path | None = None,
    validation_manifest_path: str | Path | None = None,
    smart_turn_model_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Paired candidate, public Smart Turn v3.2, and fixed-timeout evaluation."""

    manifest = Path(manifest_path or config.evaluation.test_manifest)
    validation_manifest = Path(validation_manifest_path or config.train.validation_manifest)
    target_false_cutoff_rate = config.evaluation.target_false_cutoff_rates[0]
    log_event(
        "baselines",
        "START",
        candidate=candidate_model_path,
        manifest=manifest,
        validation_manifest=validation_manifest,
        target_false_cutoff_rate=target_false_cutoff_rate,
        smart_turn_revision=SMART_TURN_REVISION,
        limit=limit or "none",
    )
    candidate = TurnDetector(candidate_model_path)
    published = SmartTurnV32Baseline(smart_turn_model_path)
    published_validation = score_manifest(
        published,
        validation_manifest,
        limit=limit,
        progress_description="baseline smart-turn-v3.2 validation calibration",
    )
    baseline_config = config.model_copy(
        update={"policy": config.policy.model_copy(update={"threshold": 0.5, "temperature": 1.0})}
    )
    published_policy, published_calibration = select_policy_from_scored(
        published_validation,
        baseline_config,
        target_false_cutoff_rate=target_false_cutoff_rate,
    )
    candidate_scored = score_manifest(
        candidate,
        manifest,
        limit=limit,
        progress_description="baseline candidate",
    )
    published_scored_raw = score_manifest(
        published,
        manifest,
        limit=limit,
        progress_description="baseline smart-turn-v3.2",
    )
    candidate_ids = [row["id"] for row in candidate_scored]
    if candidate_ids != [row["id"] for row in published_scored_raw]:
        raise RuntimeError("Candidate and baseline predictions are not aligned")
    if not candidate_scored:
        raise ValueError("Evaluation manifest is empty")

    labels = [int(row["label"]) for row in candidate_scored]
    candidate_probabilities = _probabilities(candidate_scored)
    published_default_probabilities = _probabilities(published_scored_raw)
    published_scored = _temperature_scale_scored(
        published_scored_raw,
        published_policy.temperature,
    )
    published_probabilities = _probabilities(published_scored)
    groups = [str(row["parent_id"]) for row in candidate_scored]
    candidate_policy = candidate.policy
    published_threshold = published_policy.threshold
    candidate_static = binary_classification_metrics(
        labels, candidate_probabilities, threshold=candidate_policy.threshold
    )
    published_static = binary_classification_metrics(
        labels, published_probabilities, threshold=published_threshold
    )
    published_default_static = binary_classification_metrics(
        labels, published_default_probabilities, threshold=0.5
    )

    candidate_causal = _causal_predictions(candidate_scored)
    published_causal = _causal_predictions(published_scored)
    if candidate_causal:
        fixed_timeouts = {
            str(timeout): evaluate_policy(
                candidate_causal,
                threshold=1.0,
                action_delay_ms=timeout,
                timeout_ms=timeout,
            ).as_dict()
            for timeout in (500, 800, 1_200, 1_600)
        }
        causal_deployed: dict[str, Any] | None = {
            "candidate": evaluate_policy(
                candidate_causal,
                threshold=candidate_policy.threshold,
                action_delay_ms=candidate_policy.min_silence_ms,
                timeout_ms=candidate_policy.timeout_ms,
            ).as_dict(),
            "smart_turn_v3.2": evaluate_policy(
                published_causal,
                threshold=published_threshold,
                action_delay_ms=published_policy.min_silence_ms,
                timeout_ms=published_policy.timeout_ms,
            ).as_dict(),
        }
    else:
        fixed_timeouts = {}
        causal_deployed = None

    report = {
        "manifest": str(manifest),
        "examples": len(labels),
        "candidate": {
            "model_path": str(candidate_model_path),
            "policy": candidate_policy.model_dump(mode="json"),
            "size_bytes": Path(candidate_model_path).stat().st_size,
            "model_latency": _latency(candidate_scored),
            "static": candidate_static,
            "slices": _metrics_by_slice(candidate_scored, candidate_policy.threshold),
        },
        "smart_turn_v3.2": {
            "repo": SMART_TURN_REPO,
            "revision": SMART_TURN_REVISION,
            "model_path": str(published.model_path),
            "policy": published_policy.model_dump(mode="json"),
            "validation_calibration": {
                **published_calibration,
                "manifest": str(validation_manifest),
            },
            "default_threshold_0.5_static": published_default_static,
            "size_bytes": published.model_path.stat().st_size,
            "model_latency": _latency(published_scored),
            "static": published_static,
            "slices": _metrics_by_slice(published_scored, published_threshold),
        },
        "fixed_timeout_ms": fixed_timeouts,
        "causal_deployed_policy": causal_deployed,
        "latency_contract": {
            "execution_provider": "CPUExecutionProvider",
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "end_to_end_excludes": ["audio_decode", "disk_io", "policy_silence_wait"],
        },
        "paired_tests": {
            "mcnemar": mcnemar_test(
                labels,
                candidate_probabilities,
                published_probabilities,
                threshold_a=candidate_policy.threshold,
                threshold_b=published_threshold,
            ),
            "f1_candidate_minus_smart_turn_group_bootstrap": paired_group_bootstrap_delta(
                labels,
                candidate_probabilities,
                published_probabilities,
                groups,
                metric="f1",
                threshold_a=candidate_policy.threshold,
                threshold_b=published_threshold,
                samples=config.evaluation.bootstrap_samples,
                seed=config.evaluation.seed,
            ),
        },
        "selection_split": "test",
        "baseline_policy_selection_split": "validation",
        "matched_validation_false_cutoff_budget": target_false_cutoff_rate,
        "test_policy_tuning_performed": False,
    }
    output_dir = config.evaluation.output_dir / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, output_dir / "baseline_report.json")
    for name, rows in (
        ("candidate", candidate_scored),
        ("smart_turn_v3.2", published_scored),
        ("smart_turn_v3.2_validation", published_validation),
    ):
        with (output_dir / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                serializable = {key: value for key, value in row.items() if key != "record"}
                handle.write(json.dumps(serializable))
                handle.write("\n")
    log_event(
        "baselines",
        "COMPLETE",
        examples=len(labels),
        candidate_f1=f"{candidate_static['f1']:.5f}",
        smart_turn_f1=f"{published_static['f1']:.5f}",
        output=output_dir / "baseline_report.json",
    )
    return report
