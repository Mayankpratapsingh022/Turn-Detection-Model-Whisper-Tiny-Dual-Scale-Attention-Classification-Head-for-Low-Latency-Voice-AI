from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    train_repo: str = "pipecat-ai/smart-turn-data-v3.2-train"
    test_repo: str = "pipecat-ai/smart-turn-data-v3.2-test"
    train_revision: str | None = None
    test_revision: str | None = None
    split: str = "train"
    languages: tuple[str, ...] = ("hin", "eng")
    output_dir: Path = Path("artifacts/data")
    cache_audio: bool = True
    sample_rate: int = 16_000
    max_seconds: float = 8.0
    trailing_silence_ms: int = 200
    validation_fraction: float = 0.05
    max_internal_pauses_per_clip: int = 2
    min_internal_pause_ms: int = 150
    max_internal_pause_ms: int = 1_200
    min_future_speech_ms: int = 500
    seed: int = 42
    limit: int | None = None

    @model_validator(mode="after")
    def validate_data_config(self) -> DataConfig:
        if set(self.languages) - {"hin", "eng"}:
            raise ValueError("Training languages are intentionally restricted to 'hin' and 'eng'")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.trailing_silence_ms < 100:
            raise ValueError("trailing_silence_ms must be at least 100 ms")
        if not 0 < self.min_internal_pause_ms <= self.max_internal_pause_ms:
            raise ValueError("Internal pause bounds are invalid")
        if self.max_seconds <= 0 or self.sample_rate <= 0:
            raise ValueError("Audio duration and sample rate must be positive")
        return self


class ModelConfig(StrictModel):
    base_model: str = "openai/whisper-tiny"
    base_model_revision: str | None = None
    architecture: Literal["global", "dual_scale"] = "dual_scale"
    max_seconds: float = 8.0
    sample_rate: int = 16_000
    n_mels: int = 80
    mel_frames: int = 800
    trailing_silence_ms: int = 200
    tail_seconds: float = 1.5
    attention_hidden_size: int = 128
    classifier_hidden_size: int = 256
    classifier_bottleneck_size: int = 64
    dropout: float = 0.1
    hold_loss_weight: float = 2.0
    filler_loss_weight: float = 0.15

    @model_validator(mode="after")
    def validate_feature_shape(self) -> ModelConfig:
        if self.sample_rate != 16_000 or self.n_mels != 80:
            raise ValueError("Whisper Tiny requires 16 kHz audio and 80 mel bins")
        if self.mel_frames != round(self.max_seconds * 100):
            raise ValueError("mel_frames must equal 100 frames per second")
        if self.mel_frames % 2:
            raise ValueError("mel_frames must be even for Whisper's convolution stride")
        if not 0 < self.tail_seconds <= self.max_seconds:
            raise ValueError("tail_seconds must be within the model window")
        return self


class TrainConfig(StrictModel):
    output_dir: Path = Path("artifacts/checkpoints")
    train_manifest: Path = Path("artifacts/data/train.tagged.jsonl")
    validation_manifest: Path = Path("artifacts/data/validation.tagged.jsonl")
    hard_negative_manifest: Path | None = None
    seed: int = 42
    epochs: int = 4
    physical_batch_size: int = 32
    effective_batch_size: int = 256
    num_workers: int = 8
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    freeze_encoder_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    early_stopping_patience: int = 2
    mixed_precision: Literal["bf16", "fp16", "no"] = "bf16"
    hard_negative_fraction: float = 0.30
    focused_sampling: bool = True
    include_causal_pauses: bool = True
    log_every_steps: int = 50

    @model_validator(mode="after")
    def validate_batching(self) -> TrainConfig:
        if self.physical_batch_size <= 0 or self.effective_batch_size <= 0:
            raise ValueError("Batch sizes must be positive")
        if self.effective_batch_size % self.physical_batch_size:
            raise ValueError("effective_batch_size must be divisible by physical_batch_size")
        if not 0 <= self.hard_negative_fraction < 1:
            raise ValueError("hard_negative_fraction must be in [0, 1)")
        return self

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.effective_batch_size // self.physical_batch_size


class TrackingConfig(StrictModel):
    enabled: bool = False
    project: str = "hinglish-turn-detector"
    entity: str | None = None
    run_name: str | None = None
    run_id: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"
    tags: tuple[str, ...] = ("whisper-tiny", "hinglish", "turn-detection")
    log_best_model_artifact: bool = True


class PolicyConfig(StrictModel):
    threshold: float = 0.80
    temperature: float = 1.0
    min_silence_ms: int = 200
    timeout_ms: int = 1_600
    speech_rms_threshold: float = 0.015

    @model_validator(mode="after")
    def validate_policy(self) -> PolicyConfig:
        if not 0 <= self.threshold <= 1:
            raise ValueError("Policy threshold must be in [0, 1]")
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.timeout_ms < self.min_silence_ms:
            raise ValueError("Policy timeout cannot precede min_silence_ms")
        return self


class EvaluationConfig(StrictModel):
    test_manifest: Path = Path("artifacts/data/test.tagged.jsonl")
    output_dir: Path = Path("artifacts/evaluation")
    target_false_cutoff_rates: tuple[float, ...] = (0.05, 0.10)
    thresholds: tuple[float, ...] = tuple(round(i / 20, 2) for i in range(1, 20))
    action_delays_ms: tuple[int, ...] = (200, 300, 400, 600)
    timeouts_ms: tuple[int, ...] = (600, 1_000, 1_600, 2_400)
    bootstrap_samples: int = 2_000
    robustness_limit_per_slice: int = 1_000
    seed: int = 42


class AppConfig(StrictModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def validate_audio_contract(self) -> AppConfig:
        if self.data.sample_rate != self.model.sample_rate:
            raise ValueError("Data and model sample rates must match")
        if self.data.max_seconds != self.model.max_seconds:
            raise ValueError("Data and model window durations must match")
        if self.data.trailing_silence_ms != self.model.trailing_silence_ms:
            raise ValueError("Data and model trailing-silence contracts must match")
        if self.policy.min_silence_ms < self.model.trailing_silence_ms:
            raise ValueError("Runtime silence must cover the model candidate pause")
        if any(
            delay < self.model.trailing_silence_ms for delay in self.evaluation.action_delays_ms
        ):
            raise ValueError("Evaluation action delays cannot precede the model candidate pause")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        source = Path(path)

        def merge(base: dict, override: dict) -> dict:
            result = dict(base)
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key] = merge(result[key], value)
                else:
                    result[key] = value
            return result

        with source.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        parent = payload.pop("extends", None)
        if parent:
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = source.parent / parent_path
            with parent_path.open("r", encoding="utf-8") as handle:
                base_payload = yaml.safe_load(handle) or {}
            payload = merge(base_payload, payload)
        return cls.model_validate(payload)

    def to_yaml(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False)
