from __future__ import annotations

import json
import math
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from turn_detector.config import AppConfig, EvaluationConfig
from turn_detector.data.sampling import enforce_hard_negative_fraction
from turn_detector.environment import load_project_env
from turn_detector.evaluation.metrics import (
    PausePrediction,
    binary_classification_metrics,
    operating_point,
    policy_sweep,
    tpr_at_fpr,
)
from turn_detector.modeling import create_turn_model
from turn_detector.tracking import ExperimentTracker, initialize_tracker
from turn_detector.training.dataset import TurnAudioDataset, TurnCollator


def _require_training_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Training requires `uv sync --extra train`") from exc
    return torch, get_cosine_schedule_with_warmup


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _ = _require_training_dependencies()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _autocast_context(torch: Any, device: Any, precision: str) -> Any:
    if device.type != "cuda" or precision == "no":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_loader(
    dataset: TurnAudioDataset,
    config: AppConfig,
    *,
    training: bool,
) -> Any:
    torch, _ = _require_training_dependencies()
    sampler = None
    shuffle = False
    if training and config.train.focused_sampling:
        sampling_weights = enforce_hard_negative_fraction(
            dataset.sampling_weights,
            dataset.records,
            config.train.hard_negative_fraction,
        )
        weights = torch.as_tensor(sampling_weights, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(dataset), replacement=True
        )
    elif training:
        shuffle = True
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train.physical_batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.train.num_workers > 0,
        collate_fn=TurnCollator(config.model),
        drop_last=training,
    )


def evaluate_model(
    model: Any,
    loader: Any,
    device: Any,
    precision: str,
    evaluation_config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    torch, _ = _require_training_dependencies()
    was_training = model.training
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    losses: list[float] = []
    causal_predictions: list[PausePrediction] = []
    with torch.inference_mode():
        for batch in loader:
            with _autocast_context(torch, device, precision):
                output = model(
                    input_features=batch["input_features"].to(device),
                    frame_mask=batch["frame_mask"].to(device),
                    labels=batch["labels"].to(device),
                    filler_labels=batch["filler_labels"].to(device),
                )
            losses.append(float(output["loss"].detach().cpu()))
            batch_probabilities = output["probabilities"].float().cpu().tolist()
            probabilities.extend(batch_probabilities)
            labels.extend(batch["labels"].tolist())
            for index, probability in enumerate(batch_probabilities):
                example_kind = batch["example_kinds"][index]
                label = int(batch["labels"][index])
                if example_kind == "causal_internal_pause":
                    causal_label = "hold"
                    silence_duration = int(batch["pause_durations_ms"][index] or 0)
                elif label == 1:
                    causal_label = "eot"
                    silence_duration = 10_000
                else:
                    continue
                causal_predictions.append(
                    PausePrediction(
                        id=batch["ids"][index],
                        parent_id=batch["parent_ids"][index],
                        label=causal_label,
                        probability=float(probability),
                        silence_duration_ms=silence_duration,
                        language=batch["languages"][index],
                        slice_name=batch["speech_mixes"][index],
                    )
                )
    if was_training:
        model.train()
    metrics: dict[str, Any] = binary_classification_metrics(labels, probabilities)
    tpr, threshold = tpr_at_fpr(labels, probabilities, target_fpr=0.05)
    metrics.update(
        {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "tpr_at_5pct_fpr": tpr,
            "threshold_at_5pct_fpr": threshold,
        }
    )
    if causal_predictions and evaluation_config is not None:
        sweep = policy_sweep(
            causal_predictions,
            thresholds=evaluation_config.thresholds,
            action_delays_ms=evaluation_config.action_delays_ms,
            timeouts_ms=evaluation_config.timeouts_ms,
        )
        causal_point = operating_point(sweep, max_false_cutoff_rate=0.05)
        metrics["causal_pause_predictions"] = len(causal_predictions)
        metrics["causal_operating_point_5pct_fcr"] = (
            causal_point.as_dict() if causal_point is not None else None
        )
        metrics["selection_score"] = (
            -causal_point.mean_endpoint_latency_ms if causal_point is not None else -1_000_000 + tpr
        )
    else:
        metrics["selection_score"] = tpr
    return metrics


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    output_dir: Path,
    *,
    global_step: int,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    torch, _ = _require_training_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "metrics": metrics,
        },
        output_dir / "trainer_state.pt",
    )


def _training_loss_metrics(output: dict[str, Any], optimizer: Any, epoch: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "train/loss": float(output["loss"].detach().cpu()),
        "train/epoch": epoch,
    }
    for name in ("main_loss", "filler_loss"):
        value = output.get(name)
        if value is not None:
            metrics[f"train/{name}"] = float(value.detach().cpu())
    learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if learning_rates:
        metrics["train/encoder_learning_rate"] = learning_rates[0]
    if len(learning_rates) > 1:
        metrics["train/head_learning_rate"] = learning_rates[1]
    return metrics


def _train_impl(config: AppConfig, tracker: ExperimentTracker) -> dict[str, Any]:
    torch, get_cosine_schedule_with_warmup = _require_training_dependencies()
    set_seed(config.train.seed)
    device = _device(torch)
    output_dir = config.train.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(output_dir / "resolved_config.yaml")

    train_dataset = TurnAudioDataset(
        config.train.train_manifest,
        config.model,
        augment=True,
        seed=config.train.seed,
        additional_manifest=config.train.hard_negative_manifest,
        include_causal_pauses=config.train.include_causal_pauses,
    )
    validation_dataset = TurnAudioDataset(
        config.train.validation_manifest,
        config.model,
        augment=False,
        seed=config.train.seed,
    )
    if not train_dataset.records or not validation_dataset.records:
        raise ValueError("Training and validation manifests must both be non-empty")
    train_loader = _make_loader(train_dataset, config, training=True)
    validation_loader = _make_loader(validation_dataset, config, training=False)
    if len(train_loader) == 0:
        raise ValueError("Training set is smaller than the physical batch size")

    model = create_turn_model(config.model)
    model.freeze_encoder(config.train.freeze_encoder_steps > 0)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            encoder_learning_rate=config.train.encoder_learning_rate,
            head_learning_rate=config.train.head_learning_rate,
        ),
        weight_decay=config.train.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and config.train.mixed_precision == "fp16",
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / config.train.gradient_accumulation_steps
    )
    total_steps = optimizer_steps_per_epoch * config.train.epochs
    warmup_steps = round(total_steps * config.train.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    best_score = -float("inf")
    best_saved_this_run = False
    evaluations_without_improvement = 0
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    should_stop = False
    for epoch in range(config.train.epochs):
        for batch_index, batch in enumerate(train_loader):
            if global_step == config.train.freeze_encoder_steps:
                model.freeze_encoder(False)
            with _autocast_context(torch, device, config.train.mixed_precision):
                output = model(
                    input_features=batch["input_features"].to(device),
                    frame_mask=batch["frame_mask"].to(device),
                    labels=batch["labels"].to(device),
                    filler_labels=batch["filler_labels"].to(device),
                )
                scaled_loss = output["loss"] / config.train.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index + 1
            ) % config.train.gradient_accumulation_steps == 0 or batch_index + 1 == len(
                train_loader
            )
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            should_evaluate = global_step % config.train.eval_steps == 0
            if global_step % config.train.log_every_steps == 0:
                training_metrics = _training_loss_metrics(output, optimizer, epoch)
                history.append({"step": global_step, **training_metrics})
                tracker.log(training_metrics, step=global_step, commit=not should_evaluate)

            if should_evaluate:
                metrics = evaluate_model(
                    model,
                    validation_loader,
                    device,
                    config.train.mixed_precision,
                    config.evaluation,
                )
                history.append({"step": global_step, "epoch": epoch, "validation": metrics})
                tracker.log({"validation": metrics}, step=global_step)
                score = float(metrics["selection_score"])
                if score > best_score:
                    best_score = score
                    best_saved_this_run = True
                    evaluations_without_improvement = 0
                    best_dir = output_dir / "best"
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    _save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        best_dir,
                        global_step=global_step,
                        epoch=epoch,
                        metrics=metrics,
                    )
                else:
                    evaluations_without_improvement += 1
                    if evaluations_without_improvement >= config.train.early_stopping_patience:
                        should_stop = True

            if global_step % config.train.save_steps == 0:
                _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    output_dir / f"step-{global_step}",
                    global_step=global_step,
                    epoch=epoch,
                    metrics=history[-1] if history else {},
                )
            if should_stop:
                break
        if should_stop:
            break

    final_metrics = evaluate_model(
        model,
        validation_loader,
        device,
        config.train.mixed_precision,
        config.evaluation,
    )
    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        output_dir / "final",
        global_step=global_step,
        epoch=epoch,
        metrics=final_metrics,
    )
    if not best_saved_this_run:
        best_score = float(final_metrics["selection_score"])
        best_dir = output_dir / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        _save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            best_dir,
            global_step=global_step,
            epoch=epoch,
            metrics=final_metrics,
        )
    result = {
        "device": str(device),
        "runtime_environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "global_step": global_step,
        "best_validation_selection_score": best_score,
        "final_metrics": final_metrics,
        "history": history,
    }
    (output_dir / "training_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    tracker.set_summary(
        {
            "best_validation_selection_score": best_score,
            "final": final_metrics,
            "global_step": global_step,
        }
    )
    tracker.log_model_artifact(output_dir / "best")
    return result


def train(config: AppConfig) -> dict[str, Any]:
    """Train locally or on RunPod, with optional W&B tracking from config/.env."""
    load_project_env()
    config.train.output_dir.mkdir(parents=True, exist_ok=True)
    tracker = initialize_tracker(config)
    try:
        result = _train_impl(config, tracker)
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish()
    return result
