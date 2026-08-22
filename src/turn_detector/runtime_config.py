from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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
