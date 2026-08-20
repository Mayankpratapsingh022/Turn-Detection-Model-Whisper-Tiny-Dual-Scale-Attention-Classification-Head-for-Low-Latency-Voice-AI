from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from turn_detector.audio import (
    StandardizedAudio,
    apply_corruption,
    load_audio,
    standardize_candidate_audio,
)
from turn_detector.config import ModelConfig
from turn_detector.data.records import AudioRecord, read_manifest
from turn_detector.data.sampling import compute_sampling_weights
from turn_detector.features import WhisperTurnFeatureExtractor


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Training requires `uv sync --extra train`") from exc
    return torch


try:  # Keep importing data utilities possible without the heavy training extra.
    import torch as _torch

    _DatasetBase = _torch.utils.data.Dataset
except ImportError:  # pragma: no cover - exercised in lightweight installs

    class _DatasetBase:  # type: ignore[no-redef]
        pass


class TurnAudioDataset(_DatasetBase):  # type: ignore[misc, valid-type]
    def __init__(
        self,
        manifest_path: str | Path,
        config: ModelConfig,
        *,
        augment: bool = False,
        seed: int = 42,
        additional_manifest: str | Path | None = None,
        include_causal_pauses: bool = True,
    ) -> None:
        _require_torch()
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.augment = augment
        self.seed = seed
        self.records = list(read_manifest(self.manifest_path))
        if not include_causal_pauses:
            self.records = [record for record in self.records if record.example_kind == "original"]
        if additional_manifest:
            additional = [
                record.model_copy(update={"is_hard_negative": True})
                for record in read_manifest(additional_manifest)
            ]
            self.records.extend(additional)

    def __len__(self) -> int:
        return len(self.records)

    def _augmentation_name(self, record: AudioRecord, index: int) -> str:
        digest = hashlib.sha256(f"{self.seed}:{index}:{record.id}".encode()).digest()
        selector = digest[0] / 255
        if selector < 0.50:
            return "clean"
        names = [
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
        return names[digest[1] % len(names)]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        audio_path = record.resolved_audio_path(self.manifest_path)
        waveform, sample_rate = load_audio(audio_path)
        if sample_rate != self.config.sample_rate:
            raise ValueError(f"Prepared audio must be {self.config.sample_rate} Hz: {audio_path}")
        semantic_window = waveform[-record.valid_samples :] if record.valid_samples else waveform
        if self.augment:
            corruption = self._augmentation_name(record, index)
            semantic_window = apply_corruption(
                semantic_window,
                corruption,  # type: ignore[arg-type]
                sample_rate=sample_rate,
                seed=self.seed + index,
            )
            standardized = standardize_candidate_audio(
                semantic_window,
                sample_rate,
                max_seconds=self.config.max_seconds,
                trailing_silence_ms=self.config.trailing_silence_ms,
            )
        else:
            standardized = StandardizedAudio(
                waveform=waveform,
                valid_samples=record.valid_samples,
                speech_end_sample=max(
                    0,
                    record.valid_samples
                    - round(self.config.trailing_silence_ms * sample_rate / 1_000),
                ),
                original_duration_seconds=record.duration_seconds,
            )
        filler_labels = [
            -1 if record.midfiller is None else int(record.midfiller),
            -1 if record.endfiller is None else int(record.endfiller),
        ]
        return {
            "audio": standardized,
            "label": record.label,
            "filler_labels": filler_labels,
            "id": record.id,
            "parent_id": record.parent_id,
            "language": record.language,
            "speech_mix": record.speech_mix,
            "example_kind": record.example_kind,
            "pause_duration_ms": record.pause_duration_ms,
            "is_hard_negative": record.is_hard_negative,
        }

    @property
    def sampling_weights(self) -> np.ndarray:
        return compute_sampling_weights(self.records)


class TurnCollator:
    def __init__(self, config: ModelConfig) -> None:
        self.extractor = WhisperTurnFeatureExtractor(config)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        torch = _require_torch()
        features = self.extractor([example["audio"] for example in examples], return_tensors="pt")
        return {
            "input_features": features.input_features,
            "frame_mask": features.frame_mask,
            "labels": torch.tensor([example["label"] for example in examples], dtype=torch.long),
            "filler_labels": torch.tensor(
                [example["filler_labels"] for example in examples], dtype=torch.long
            ),
            "ids": [example["id"] for example in examples],
            "parent_ids": [example["parent_id"] for example in examples],
            "languages": [example["language"] for example in examples],
            "speech_mixes": [example["speech_mix"] for example in examples],
            "example_kinds": [example["example_kind"] for example in examples],
            "pause_durations_ms": [example["pause_duration_ms"] for example in examples],
            "hard_negatives": [example["is_hard_negative"] for example in examples],
        }
