from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.json import JSON

from turn_detector.config import AppConfig
from turn_detector.environment import load_project_env

app = typer.Typer(
    name="turn-detector",
    help="Train and evaluate a tiny Hindi/Hinglish audio turn detector.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Audit and prepare Smart Turn audio.", no_args_is_help=True)
app.add_typer(data_app, name="data")
console = Console()


ConfigPath = Annotated[
    Path,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML configuration."),
]


def _show(payload: object) -> None:
    console.print(JSON.from_data(payload))


@app.command("validate-config")
def validate_config(config_path: ConfigPath = Path("configs/default.yaml")) -> None:
    """Validate all config fields and print the resolved values."""
    _show(AppConfig.from_yaml(config_path).model_dump(mode="json"))


@data_app.command("prepare")
def data_prepare(
    config_path: ConfigPath = Path("configs/default.yaml"),
    train_only: Annotated[
        bool, typer.Option(help="Skip preparation of the companion test set.")
    ] = False,
    limit: Annotated[
        int | None, typer.Option(help="Accepted parent utterances per repository.")
    ] = None,
) -> None:
    """Prepare duplicate-safe Hindi/English train, validation, and test manifests."""
    load_project_env()
    from turn_detector.data.prepare import prepare_dataset, prepare_train_and_test

    config = AppConfig.from_yaml(config_path)
    if limit is not None:
        config = config.model_copy(update={"data": config.data.model_copy(update={"limit": limit})})
    summary = (
        prepare_dataset(config.data, repo_id=config.data.train_repo)
        if train_only
        else prepare_train_and_test(config.data)
    )
    _show(summary)


@data_app.command("audit")
def data_audit(
    config_path: ConfigPath = Path("configs/default.yaml"),
    limit: Annotated[int, typer.Option(help="Relevant clips to inspect.")] = 1_000,
) -> None:
    """Run a bounded streaming quality audit without a full corpus download."""
    load_project_env()
    from turn_detector.data.prepare import prepare_dataset

    config = AppConfig.from_yaml(config_path)
    audit_data = config.data.model_copy(
        update={
            "output_dir": config.data.output_dir / "audit_sample",
            "cache_audio": False,
            "limit": limit,
            "max_internal_pauses_per_clip": 0,
        }
    )
    _show(prepare_dataset(audit_data, repo_id=audit_data.train_repo))


@data_app.command("tag-hinglish")
def tag_hinglish(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    model: Annotated[str, typer.Option(help="faster-whisper checkpoint.")] = "large-v3",
    device: Annotated[str, typer.Option()] = "cuda",
    compute_type: Annotated[str, typer.Option()] = "float16",
    limit: Annotated[int | None, typer.Option()] = None,
    checkpoint_every: Annotated[int, typer.Option(min=1)] = 250,
) -> None:
    """Add high-confidence Hinglish/Hindi/English tags using offline ASR."""
    load_project_env()
    from turn_detector.data.hinglish import tag_manifest_with_asr

    _show(
        tag_manifest_with_asr(
            manifest,
            output,
            model_name=model,
            device=device,
            compute_type=compute_type,
            limit=limit,
            checkpoint_every=checkpoint_every,
        )
    )


@data_app.command("tag-all")
def tag_all_hinglish(
    data_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("artifacts/data"),
    model: Annotated[str, typer.Option(help="faster-whisper checkpoint.")] = "large-v3",
    device: Annotated[str, typer.Option()] = "cuda",
    compute_type: Annotated[str, typer.Option()] = "float16",
    checkpoint_every: Annotated[int, typer.Option(min=1)] = 250,
) -> None:
    """Optionally audit train/validation/test in one resumable ASR session."""
    load_project_env()
    from turn_detector.data.hinglish import tag_prepared_splits_with_asr

    _show(
        tag_prepared_splits_with_asr(
            data_dir,
            model_name=model,
            device=device,
            compute_type=compute_type,
            checkpoint_every=checkpoint_every,
        )
    )


@data_app.command("summary")
def data_summary(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Summarize a prepared manifest."""
    from turn_detector.data.prepare import summarize_records
    from turn_detector.data.records import read_manifest

    _show(summarize_records(read_manifest(manifest)))


@app.command("train")
def train_command(config_path: ConfigPath = Path("configs/default.yaml")) -> None:
    """Train the configured global or dual-scale model."""
    from turn_detector.training.trainer import train

    _show(train(AppConfig.from_yaml(config_path)))


@app.command("cache-assets")
def cache_assets_command(
    config_path: ConfigPath = Path("configs/runpod.yaml"),
    datasets: Annotated[
        bool, typer.Option("--datasets/--no-datasets", help="Cache train and test datasets.")
    ] = True,
    model: Annotated[
        bool, typer.Option("--model/--no-model", help="Cache the Whisper Tiny base model.")
    ] = True,
    asr: Annotated[
        bool,
        typer.Option("--asr/--no-asr", help="Optionally cache faster-whisper for ASR analysis."),
    ] = False,
    cache_dir: Annotated[Path | None, typer.Option(file_okay=False)] = None,
    manifest: Annotated[Path, typer.Option(dir_okay=False)] = Path("artifacts/cache_manifest.json"),
) -> None:
    """Download the configured HF datasets and models into persistent cache."""
    load_project_env()
    from turn_detector.hub_cache import cache_huggingface_assets

    _show(
        cache_huggingface_assets(
            AppConfig.from_yaml(config_path),
            include_datasets=datasets,
            include_model=model,
            include_asr=asr,
            cache_dir=cache_dir,
            manifest_path=manifest,
        )
    )


@app.command("package-model")
def package_model_command(
    checkpoint: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    export_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "artifacts/export"
    ),
    output: Annotated[Path, typer.Option(file_okay=False)] = Path("artifacts/release"),
    evaluation_dir: Annotated[Path | None, typer.Option(file_okay=False)] = Path(
        "artifacts/evaluation"
    ),
    model_card: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Stage weights, ONNX, policy, reports, model card, hashes, and license for the Hub."""
    from turn_detector.publishing import stage_model_release

    _show(
        stage_model_release(
            checkpoint,
            export_dir,
            output,
            evaluation_dir=evaluation_dir,
            model_card=model_card,
        )
    )


@app.command("pin-config")
def pin_config_command(
    config_path: ConfigPath = Path("configs/runpod.yaml"),
    cache_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/cache_manifest.json"
    ),
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "artifacts/configs/runpod.pinned.yaml"
    ),
) -> None:
    """Pin dataset and base-model revisions to a completed cache manifest."""
    from turn_detector.hub_cache import pin_config_to_cached_revisions

    _show(pin_config_to_cached_revisions(AppConfig.from_yaml(config_path), cache_manifest, output))


@app.command("push-model")
def push_model_command(
    folder: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("artifacts/release"),
    repo_id: Annotated[
        str | None, typer.Option(help="Hugging Face owner/model; defaults to HF_MODEL_REPO.")
    ] = None,
    private: Annotated[bool, typer.Option("--private/--public")] = True,
    revision: Annotated[str, typer.Option()] = "main",
    create_pr: Annotated[bool, typer.Option()] = False,
    acknowledge_source_license_review: Annotated[
        bool,
        typer.Option(
            "--acknowledge-source-license-review",
            help="Confirm that upstream dataset terms were reviewed before uploading weights.",
        ),
    ] = False,
) -> None:
    """Explicitly upload a packaged release to a private HF model repo by default."""
    load_project_env()
    from turn_detector.publishing import push_model_to_hub

    selected_repo = repo_id or os.getenv("HF_MODEL_REPO")
    if not selected_repo:
        raise typer.BadParameter("Set --repo-id or HF_MODEL_REPO in .env", param_hint="repo-id")
    _show(
        push_model_to_hub(
            folder,
            selected_repo,
            private=private,
            revision=revision,
            create_pr=create_pr,
            acknowledge_source_license_review=acknowledge_source_license_review,
        )
    )


@app.command("mine-hard-negatives")
def mine_hard_negatives_command(
    model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "artifacts/data/hard_negatives.jsonl"
    ),
    config_path: ConfigPath = Path("configs/default.yaml"),
) -> None:
    """Mine the incomplete examples most likely to be falsely cut off."""
    from turn_detector.training.hard_negatives import mine_hard_negatives

    config = AppConfig.from_yaml(config_path)
    _show(
        mine_hard_negatives(
            model_path,
            manifest,
            output,
            model_config=config.model,
        )
    )


@app.command("evaluate")
def evaluate_command(
    model_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    config_path: ConfigPath = Path("configs/default.yaml"),
    manifest: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    limit: Annotated[int | None, typer.Option()] = None,
    robustness: Annotated[bool, typer.Option("--robustness/--no-robustness")] = True,
    output_dir: Annotated[Path | None, typer.Option(file_okay=False)] = None,
) -> None:
    """Run static, causal, slice, calibration, robustness, and latency evaluation."""
    from turn_detector.evaluation.evaluator import evaluate

    config = AppConfig.from_yaml(config_path)
    if output_dir is not None:
        config = config.model_copy(
            update={"evaluation": config.evaluation.model_copy(update={"output_dir": output_dir})}
        )
    _show(
        evaluate(
            model_path,
            config,
            manifest_path=manifest,
            limit=limit,
            run_robustness=robustness,
        )
    )


@app.command("calibrate")
def calibrate_command(
    model_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    config_path: ConfigPath = Path("configs/default.yaml"),
    manifest: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
    target_false_cutoff_rate: Annotated[
        float, typer.Option(min=0.0, max=0.5, help="Validation interruption budget.")
    ] = 0.05,
    limit: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Fit temperature and the endpoint policy on validation data only."""
    from turn_detector.evaluation.calibration import calibrate

    _show(
        calibrate(
            model_path,
            AppConfig.from_yaml(config_path),
            manifest_path=manifest,
            output_path=output,
            target_false_cutoff_rate=target_false_cutoff_rate,
            limit=limit,
        )
    )


@app.command("compare-baselines")
def compare_baselines_command(
    model_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    config_path: ConfigPath = Path("configs/default.yaml"),
    manifest: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    validation_manifest: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Validation-only split used to calibrate the public baseline policy.",
        ),
    ] = None,
    smart_turn_model: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    limit: Annotated[int | None, typer.Option()] = None,
    output_dir: Annotated[Path | None, typer.Option(file_okay=False)] = None,
) -> None:
    """Compare against pinned Smart Turn v3.2 and fixed VAD timeouts."""
    from turn_detector.evaluation.baselines import compare_baselines

    config = AppConfig.from_yaml(config_path)
    if output_dir is not None:
        config = config.model_copy(
            update={"evaluation": config.evaluation.model_copy(update={"output_dir": output_dir})}
        )
    _show(
        compare_baselines(
            model_path,
            config,
            manifest_path=manifest,
            validation_manifest_path=validation_manifest,
            smart_turn_model_path=smart_turn_model,
            limit=limit,
        )
    )


@app.command("benchmark")
def benchmark_command(
    model_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    audio_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    iterations: Annotated[int, typer.Option(min=10)] = 1_000,
) -> None:
    """Measure warm model and end-to-end latency on one audio file."""
    from turn_detector.audio import load_audio
    from turn_detector.evaluation.evaluator import benchmark_latency
    from turn_detector.inference import TurnDetector

    audio, sample_rate = load_audio(audio_path)
    _show(benchmark_latency(TurnDetector(model_path), audio, sample_rate, iterations=iterations))


@app.command("export")
def export_command(
    checkpoint: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "artifacts/export/hinglish-turn.onnx"
    ),
    config_path: ConfigPath = Path("configs/default.yaml"),
    quantize: Annotated[bool, typer.Option("--quantize/--no-quantize")] = True,
    static_quantization: Annotated[
        bool, typer.Option("--static/--dynamic", help="Use held-out activation calibration.")
    ] = True,
    calibration_manifest: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    calibration_samples: Annotated[int, typer.Option(min=32)] = 1_024,
) -> None:
    """Export and verify FP32 and optional INT8 ONNX models."""
    from turn_detector.export import export_onnx

    config = AppConfig.from_yaml(config_path)
    selected_calibration = None
    if quantize and static_quantization:
        selected_calibration = calibration_manifest or config.train.validation_manifest
    _show(
        export_onnx(
            checkpoint,
            output,
            policy=config.policy,
            quantize=quantize,
            calibration_manifest=selected_calibration,
            calibration_samples=calibration_samples,
        )
    )


@app.command("predict")
def predict_command(
    model_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    audio_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Score one audio file."""
    from turn_detector.audio import load_audio
    from turn_detector.inference import TurnDetector

    audio, sample_rate = load_audio(audio_path)
    _show(TurnDetector(model_path).score(audio, sample_rate).as_dict())


@app.command("demo")
def demo_command(
    model_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    share: Annotated[bool, typer.Option()] = False,
    server_port: Annotated[int, typer.Option()] = 7860,
) -> None:
    """Launch the Gradio microphone and upload demo."""
    from turn_detector.demo import build_demo

    build_demo(model_path).launch(share=share, server_port=server_port)


@app.command("show-report")
def show_report(
    report_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Pretty-print a generated JSON report."""
    _show(json.loads(report_path.read_text()))


if __name__ == "__main__":
    app()
