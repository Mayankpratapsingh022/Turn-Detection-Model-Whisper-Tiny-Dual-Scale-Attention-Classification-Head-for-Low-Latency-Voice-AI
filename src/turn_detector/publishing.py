from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from turn_detector.progress import log_event


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _first_existing(paths: tuple[Path, ...], description: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    choices = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Missing {description}; checked: {choices}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _number(value: Any, *, digits: int = 4, suffix: str = "") -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "Not available"
    return f"{float(value):.{digits}f}{suffix}"


def _integer(value: Any) -> str:
    return (
        f"{int(value):,}"
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        else "Not available"
    )


def _render_result_model_card(
    checkpoint_config: Path,
    export_report_path: Path,
    policy_path: Path,
    *,
    evaluation_dir: Path | None,
) -> str:
    model_payload = _read_json(checkpoint_config)
    model_config = model_payload.get("turn_detector_config", model_payload)
    export = _read_json(export_report_path)
    policy = _read_json(policy_path)
    evaluation = (
        _read_json(evaluation_dir / "evaluation_report.json") if evaluation_dir is not None else {}
    )
    baseline = (
        _read_json(evaluation_dir / "baselines" / "baseline_report.json")
        if evaluation_dir is not None
        else {}
    )
    calibration = _read_json(export_report_path.with_name("calibration_report.json"))

    static = evaluation.get("static", {})
    causal = evaluation.get("causal", {}).get("deployed_policy") or {}
    latency = evaluation.get("latency", {})
    runtime = evaluation.get("runtime_environment", {})
    candidate_baseline = baseline.get("candidate", {}).get("static", {})
    public_baseline = baseline.get("smart_turn_v3.2", {}).get("static", {})
    paired = baseline.get("paired_tests", {}).get(
        "f1_candidate_minus_smart_turn_group_bootstrap", {}
    )

    metric_rows = [
        ("Examples", _integer(static.get("count"))),
        ("F1", _number(static.get("f1"))),
        ("Balanced accuracy", _number(static.get("balanced_accuracy"))),
        ("AUROC", _number(static.get("auroc"))),
        ("Average precision", _number(static.get("average_precision"))),
        ("False-cutoff rate", _number(static.get("false_cutoff_rate"))),
        ("False-hold rate", _number(static.get("false_hold_rate"))),
        ("Expected calibration error", _number(static.get("ece"))),
    ]
    metric_table = "\n".join(f"| {name} | {value} |" for name, value in metric_rows)

    slice_rows: list[str] = []
    slices = evaluation.get("slices", {})
    for name in (
        "language:hin",
        "language:eng",
        "filler:mid",
        "filler:end",
        "filler:any",
        "hard:incomplete_filler",
        "kind:causal_internal_pause",
        "kind:original",
    ):
        values = slices.get(name)
        if not isinstance(values, dict):
            continue
        slice_rows.append(
            f"| `{name}` | {_integer(values.get('count'))} | "
            f"{_number(values.get('f1'))} | {_number(values.get('false_cutoff_rate'))} |"
        )
    if not slice_rows:
        slice_rows.append("| Not available | — | — | — |")

    robustness_rows: list[str] = []
    robustness = evaluation.get("robustness", {})
    for name, values in robustness.items():
        if not isinstance(values, dict):
            continue
        robustness_rows.append(
            f"| `{name}` | {_number(values.get('f1'))} | "
            f"{_number(values.get('false_cutoff_rate'))} |"
        )
    if not robustness_rows:
        robustness_rows.append("| Not available | — | — |")

    model_size_mb = (
        float(export["int8_size_bytes"]) / (1024 * 1024)
        if isinstance(export.get("int8_size_bytes"), (int, float))
        else None
    )
    baseline_rows = [
        ("Candidate", candidate_baseline.get("f1")),
        ("Smart Turn v3.2", public_baseline.get("f1")),
    ]
    baseline_table = "\n".join(f"| {name} | {_number(value)} |" for name, value in baseline_rows)

    return f"""---
language:
- hi
- en
pipeline_tag: audio-classification
tags:
- turn-detection
- semantic-vad
- hindi
- hinglish-target-domain
- fillers
- onnx
license: other
license_name: pending-source-data-review
---

# Whisper Tiny Dual-Scale Turn Detector

A compact audio-only classifier that predicts whether a speaker has completed their turn or is
holding the floor. It is designed to run after a lightweight VAD observes a candidate pause. This
model does not require ASR or text at inference time.

This card is generated from the packaged run artifacts. Missing results are shown as **Not
available** instead of being estimated or hand-entered.

## Model details

- Base encoder: `{model_config.get("base_model", "openai/whisper-tiny")}`
- Architecture: `{model_config.get("architecture", "dual_scale")}`
- Input: {model_config.get("sample_rate", 16_000)} Hz mono audio, last
  {_number(model_config.get("max_seconds"), digits=1, suffix=" s")}
- Output: calibrated probability of end-of-turn / `COMPLETE`
- INT8 ONNX size: {_number(model_size_mb, digits=2, suffix=" MiB")}
- INT8 method: `{export.get("quantization_method", "Not available")}`
- INT8 maximum probability difference from FP32: {_number(export.get("int8_max_probability_difference"), digits=6)}
- Export parity target passed: `{export.get("meets_int8_probability_parity_target", "Not available")}`

The encoder feeds masked global attention pooling and a recent-window attention/mean/max branch.
The fused representation predicts turn completion. Separate mid-filler and end-filler auxiliary
outputs supervise training but are not required by the exported inference graph.

## Test results

The candidate temperature and endpoint policy are selected on validation data and frozen before
test evaluation.

| Metric | Test value |
|---|---:|
{metric_table}

### Causal turn policy

- Threshold: {_number(policy.get("threshold"))}
- Temperature: {_number(policy.get("temperature"))}
- Minimum silence: {_integer(policy.get("min_silence_ms"))} ms
- Fallback timeout: {_integer(policy.get("timeout_ms"))} ms
- Turn-level false-cutoff rate: {_number(causal.get("false_cutoff_rate"))}
- Mean endpoint latency: {_number(causal.get("mean_endpoint_latency_ms"), digits=1, suffix=" ms")}
- P95 endpoint latency: {_number(causal.get("p95_endpoint_latency_ms"), digits=1, suffix=" ms")}
- Calibration split: `{calibration.get("selection_split", "Not available")}`
- Test policy tuning performed: `{evaluation.get("test_policy_tuning_performed", False)}`

### Important slices

| Slice | Count | F1 | False-cutoff rate |
|---|---:|---:|---:|
{chr(10).join(slice_rows)}

### Robustness

The robustness subset is deterministically stratified across language, label, original/causal
examples, filler type, and synthetic status. Each corruption uses the same selected records.

| Condition | F1 | False-cutoff rate |
|---|---:|---:|
{chr(10).join(robustness_rows)}

### CPU latency

- Model-only P50/P95/P99: {_number(latency.get("model_p50_ms"), digits=2)} / {_number(latency.get("model_p95_ms"), digits=2)} / {_number(latency.get("model_p99_ms"), digits=2)} ms
- End-to-end P50/P95/P99: {_number(latency.get("end_to_end_p50_ms"), digits=2)} / {_number(latency.get("end_to_end_p95_ms"), digits=2)} / {_number(latency.get("end_to_end_p99_ms"), digits=2)} ms
- Runtime: `{runtime.get("platform", "Not available")}`, {runtime.get("logical_cpu_count", "unknown")} logical CPUs

End-to-end latency includes waveform standardization, log-Mel extraction, probability calibration,
and ONNX inference. It excludes audio decoding/disk I/O and the configured silence wait.

### Public baseline

Smart Turn v3.2 is pinned by revision. Its temperature, threshold, action delay, and timeout are
selected on validation data under the same configured false-cutoff budget as the candidate, then
both policies are frozen for the paired test comparison.

| Model | Test F1 |
|---|---:|
{baseline_table}

- Candidate minus Smart Turn F1: {_number(paired.get("delta"))}
- 95% group-bootstrap interval: [{_number(paired.get("ci_low"))}, {_number(paired.get("ci_high"))}]
- Matched validation false-cutoff budget: {_number(baseline.get("matched_validation_false_cutoff_budget"))}

## Training and evaluation data

Training and in-domain evaluation use only the Hindi (`hin`) and English (`eng`) subsets of the
provided Smart Turn v3.2 train/test dataset family. `endpoint_bool` supplies the main completion
label; `midfiller` and `endfiller` supply auxiliary supervision. Causal internal-pause examples and
mined hard negatives teach the model not to interrupt a speaker who is likely to continue.

The test split remains separate from threshold, temperature, policy, and baseline selection.
Reported confidence intervals use parent-turn group bootstrapping. Robustness results cover
telephone filtering, mu-law, additive noise, speed changes, low gain, clipping, and reverb.

## Limitations and honest scope

- The source has Hindi/English metadata but no human-verified Hinglish or code-switch label.
  Consequently, this release does **not** claim a measured Hinglish-specific test score. Hindi and
  filler-focused results are relevant proxies, not a Hinglish ground-truth benchmark.
- English rows are not guaranteed to be exclusively Indian English.
- Some source audio is synthetic, and the source does not provide reliable speaker identities.
- The detector uses audio only; it cannot use conversation history, transcript semantics, gaze,
  or dialog state.
- It is not a backchannel, barge-in, speaker-diarization, or safety classifier.
- Review the upstream dataset terms before distributing derived weights. Source audio is not
  included in this package.

## Minimal inference

```python
from turn_detector.inference import TurnDetector
from turn_detector.audio import load_audio

detector = TurnDetector("hinglish-turn.int8.onnx")
audio, sample_rate = load_audio("candidate_pause.wav")
prediction = detector.score(audio, sample_rate)
print(prediction.probability, prediction.decision)
```

See `evaluation/evaluation_report.json`, `evaluation/baselines/baseline_report.json`,
`calibration_report.json`, and `export_report.json` for the complete machine-readable results.
"""


def stage_model_release(
    checkpoint_dir: Path,
    export_dir: Path,
    output_dir: Path,
    *,
    evaluation_dir: Path | None = None,
    model_card: Path | None = None,
) -> dict[str, Any]:
    """Assemble a self-contained Hub folder without optimizer state or source audio."""
    log_event(
        "package",
        "START",
        checkpoint=checkpoint_dir,
        export_dir=export_dir,
        evaluation_dir=evaluation_dir,
        output=output_dir,
    )
    repository_root = Path(__file__).resolve().parents[2]
    checkpoint_config = _first_existing(
        (checkpoint_dir / "turn_detector_config.json",), "checkpoint model config"
    )
    checkpoint_weights = _first_existing(
        (checkpoint_dir / "model.safetensors", checkpoint_dir / "pytorch_model.bin"),
        "checkpoint model weights",
    )
    int8_model = _first_existing(tuple(sorted(export_dir.glob("*.int8.onnx"))), "INT8 ONNX model")
    export_report = _first_existing((export_dir / "export_report.json",), "export report")
    policy = _first_existing((export_dir / "policy.json",), "calibrated/default policy")
    selected_license = repository_root / "LICENSE"
    required_files = [selected_license]
    if model_card is not None:
        required_files.append(model_card)
    for required in required_files:
        if not required.exists():
            raise FileNotFoundError(required)

    sources: list[tuple[Path, Path]] = [
        (checkpoint_config, Path("turn_detector_config.json")),
        (checkpoint_weights, Path(checkpoint_weights.name)),
        (int8_model, Path(int8_model.name)),
        (export_report, Path("export_report.json")),
        (policy, Path("policy.json")),
        (selected_license, Path("LICENSE")),
    ]
    if model_card is not None:
        sources.append((model_card, Path("README.md")))
    fp32_models = sorted(export_dir.glob("*.onnx"))
    for model in fp32_models:
        if model != int8_model:
            sources.append((model, Path(model.name)))
    optional_reports = [
        export_dir / "calibration_report.json",
        checkpoint_dir.parent / "training_report.json",
        checkpoint_dir.parent / "resolved_config.yaml",
    ]
    for report in optional_reports:
        if report.exists():
            sources.append((report, Path(report.name)))
    if evaluation_dir is not None and evaluation_dir.exists():
        allowed_suffixes = {".json", ".csv", ".png", ".md"}
        for source in sorted(evaluation_dir.rglob("*")):
            if source.is_file() and source.suffix.lower() in allowed_suffixes:
                relative = source.relative_to(evaluation_dir)
                sources.append((source, Path("evaluation") / relative))

    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    seen_destinations: set[Path] = set()
    for source, relative in sources:
        if relative in seen_destinations:
            continue
        seen_destinations.add(relative)
        destination = output_dir / relative
        _copy(source, destination)
        copied.append(
            {
                "path": relative.as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )

    if model_card is None:
        generated_card = output_dir / "README.md"
        generated_card.write_text(
            _render_result_model_card(
                checkpoint_config,
                export_report,
                policy,
                evaluation_dir=evaluation_dir,
            ),
            encoding="utf-8",
        )
        copied.append(
            {
                "path": "README.md",
                "size_bytes": generated_card.stat().st_size,
                "sha256": _sha256(generated_card),
            }
        )

    manifest = {
        "format_version": 1,
        "checkpoint_source": str(checkpoint_dir),
        "export_source": str(export_dir),
        "files": copied,
        "excluded": ["trainer_state.pt", "source_audio", ".env"],
    }
    manifest_path = output_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {"release_dir": str(output_dir), **manifest}
    log_event(
        "package",
        "COMPLETE",
        files=len(copied),
        total_bytes=sum(int(entry["size_bytes"]) for entry in copied),
        manifest=manifest_path,
    )
    return result


def push_model_to_hub(
    folder: Path,
    repo_id: str,
    *,
    private: bool = True,
    revision: str = "main",
    create_pr: bool = False,
    commit_message: str = "Upload trained Hinglish turn detector",
    acknowledge_source_license_review: bool = False,
) -> dict[str, Any]:
    """Upload a staged release. This is never called automatically by training."""
    if not acknowledge_source_license_review:
        raise ValueError(
            "Publishing requires --acknowledge-source-license-review because the upstream "
            "dataset card does not declare an explicit license."
        )
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Add it to .env before publishing.")
    manifest_path = folder / "release_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Package the model first; release_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allow_patterns = [entry["path"] for entry in manifest["files"]]
    allow_patterns.append("release_manifest.json")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Hub publishing requires `uv sync --extra hub`") from exc
    api = HfApi(token=token)
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        create_pr=create_pr,
        commit_message=commit_message,
        allow_patterns=allow_patterns,
        token=token,
    )
    return {
        "repo_id": repo_id,
        "repo_url": str(repo_url),
        "commit_url": getattr(commit, "commit_url", None),
        "revision": revision,
        "private": private,
        "uploaded_files": allow_patterns,
    }
