from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from turn_detector.audio import StandardizedAudio
from turn_detector.runtime_config import ModelConfig


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    input_features: Any
    frame_mask: Any


class WhisperTurnFeatureExtractor:
    def __init__(self, config: ModelConfig) -> None:
        try:
            from transformers import WhisperFeatureExtractor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Feature extraction requires `uv sync --extra train`") from exc
        self.config = config
        self.extractor = WhisperFeatureExtractor(
            feature_size=config.n_mels,
            sampling_rate=config.sample_rate,
            hop_length=160,
            chunk_length=int(config.max_seconds),
            n_fft=400,
            padding_value=0.0,
            return_attention_mask=False,
        )

    def __call__(
        self,
        samples: StandardizedAudio | list[StandardizedAudio],
        *,
        return_tensors: str = "pt",
    ) -> FeatureBatch:
        batch = samples if isinstance(samples, list) else [samples]
        waveforms = [item.waveform for item in batch]
        extracted = self.extractor(
            waveforms,
            sampling_rate=self.config.sample_rate,
            return_tensors=return_tensors,
            padding="max_length",
            max_length=round(self.config.max_seconds * self.config.sample_rate),
            truncation=True,
            do_normalize=True,
        )
        masks = np.stack([item.mel_frame_mask(n_frames=self.config.mel_frames) for item in batch])
        frame_mask: Any
        if return_tensors == "pt":
            import torch

            frame_mask = torch.from_numpy(masks)
        else:
            frame_mask = masks
        return FeatureBatch(input_features=extracted.input_features, frame_mask=frame_mask)
