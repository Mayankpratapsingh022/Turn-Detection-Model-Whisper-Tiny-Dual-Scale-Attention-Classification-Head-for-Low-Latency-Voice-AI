from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from turn_detector.config import ModelConfig


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Model code requires `uv sync --extra train`") from exc
    return torch, nn, functional


class MaskedAttentionPool:  # constructed dynamically as nn.Module below
    pass


def _build_masked_attention_pool(hidden_size: int, attention_size: int) -> Any:
    torch, nn, _ = _require_torch()

    class _MaskedAttentionPool(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.scorer = nn.Sequential(
                nn.Linear(hidden_size, attention_size),
                nn.Tanh(),
                nn.Linear(attention_size, 1),
            )

        def forward(self, hidden_states: Any, mask: Any) -> Any:
            valid = mask.to(dtype=torch.bool)
            scores = self.scorer(hidden_states).squeeze(-1)
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            all_padding = ~valid.any(dim=1)
            if all_padding.any():
                scores = scores.clone()
                scores[all_padding, -1] = 0.0
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = torch.sum(hidden_states * weights, dim=1)
            return pooled.masked_fill(all_padding.unsqueeze(-1), 0.0)

    return _MaskedAttentionPool()


def _build_model_class() -> type[Any]:
    torch, nn, functional = _require_torch()

    class DualScaleTurnDetector(nn.Module):  # type: ignore[name-defined]
        def __init__(
            self,
            config: ModelConfig,
            *,
            encoder: Any | None = None,
            encoder_config: dict[str, Any] | None = None,
        ) -> None:
            super().__init__()
            self.turn_config = config
            if encoder is None:
                from transformers import WhisperConfig, WhisperModel
                from transformers.models.whisper.modeling_whisper import WhisperEncoder

                if encoder_config is not None:
                    whisper_config = WhisperConfig.from_dict(encoder_config)
                    encoder = WhisperEncoder(whisper_config)
                else:
                    base = WhisperModel.from_pretrained(
                        config.base_model,
                        revision=config.base_model_revision,
                    )
                    encoder = base.encoder
                    target_positions = config.mel_frames // 2
                    if encoder.config.max_source_positions != target_positions:
                        old_positions = encoder.embed_positions.weight.detach()[
                            :target_positions
                        ].clone()
                        encoder.config.max_source_positions = target_positions
                        encoder.max_source_positions = target_positions
                        replacement = nn.Embedding(target_positions, encoder.config.d_model)
                        replacement.weight.data.copy_(old_positions)
                        replacement.weight.requires_grad_(False)
                        encoder.embed_positions = replacement
            self.encoder = encoder
            hidden_size = int(self.encoder.config.d_model)
            self.hidden_size = hidden_size
            self.global_pool = _build_masked_attention_pool(
                hidden_size, config.attention_hidden_size
            )
            self.tail_pool = _build_masked_attention_pool(hidden_size, config.attention_hidden_size)
            multiplier = 1 if config.architecture == "global" else 4
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size * multiplier),
                nn.Linear(hidden_size * multiplier, config.classifier_hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.classifier_hidden_size, config.classifier_bottleneck_size),
                nn.GELU(),
                nn.Linear(config.classifier_bottleneck_size, 1),
            )
            self.filler_head = nn.Linear(hidden_size * 2, 1)

        @staticmethod
        def _encoder_mask(frame_mask: Any, target_length: int) -> Any:
            mask = frame_mask.to(dtype=torch.float32).unsqueeze(1)
            mask = functional.max_pool1d(mask, kernel_size=2, stride=2)
            if mask.shape[-1] != target_length:
                mask = functional.interpolate(mask, size=target_length, mode="nearest")
            return mask.squeeze(1).to(dtype=torch.bool)

        @staticmethod
        def _tail_mask(valid_mask: Any, tail_frames: int) -> Any:
            positions = torch.arange(valid_mask.shape[1], device=valid_mask.device).unsqueeze(0)
            last_valid = positions.masked_fill(~valid_mask, -1).max(dim=1).values
            first_tail = (last_valid - tail_frames + 1).clamp(min=0).unsqueeze(1)
            return valid_mask & (positions >= first_tail)

        @staticmethod
        def _masked_mean(hidden_states: Any, mask: Any) -> Any:
            weights = mask.to(hidden_states.dtype).unsqueeze(-1)
            return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

        @staticmethod
        def _masked_max(hidden_states: Any, mask: Any) -> Any:
            values = hidden_states.masked_fill(
                ~mask.unsqueeze(-1), torch.finfo(hidden_states.dtype).min
            )
            result = values.max(dim=1).values
            return result.masked_fill(~mask.any(dim=1).unsqueeze(-1), 0.0)

        def forward(
            self,
            input_features: Any,
            frame_mask: Any,
            labels: Any | None = None,
            filler_labels: Any | None = None,
        ) -> dict[str, Any]:
            encoder_output = self.encoder(input_features=input_features, return_dict=True)
            hidden_states = encoder_output.last_hidden_state
            valid_mask = self._encoder_mask(frame_mask, hidden_states.shape[1])
            global_embedding = self.global_pool(hidden_states, valid_mask)
            tail_frames = max(1, round(self.turn_config.tail_seconds * 50))
            tail_mask = self._tail_mask(valid_mask, tail_frames)
            tail_attention = self.tail_pool(hidden_states, tail_mask)
            if self.turn_config.architecture == "global":
                fused = global_embedding
            else:
                tail_mean = self._masked_mean(hidden_states, tail_mask)
                tail_max = self._masked_max(hidden_states, tail_mask)
                fused = torch.cat([global_embedding, tail_attention, tail_mean, tail_max], dim=-1)
            logits = self.classifier(fused).squeeze(-1)
            filler_logits = self.filler_head(
                torch.cat([global_embedding, tail_attention], dim=-1)
            ).squeeze(-1)
            result: dict[str, Any] = {
                "logits": logits,
                "probabilities": torch.sigmoid(logits),
                "filler_logits": filler_logits,
            }
            if labels is not None:
                labels_float = labels.to(dtype=logits.dtype)
                per_sample = functional.binary_cross_entropy_with_logits(
                    logits, labels_float, reduction="none"
                )
                sample_weights = torch.where(
                    labels_float < 0.5,
                    torch.full_like(labels_float, self.turn_config.hold_loss_weight),
                    torch.ones_like(labels_float),
                )
                main_loss = (per_sample * sample_weights).mean()
                loss = main_loss
                filler_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
                if filler_labels is not None:
                    filler_valid = filler_labels >= 0
                    if filler_valid.any():
                        filler_loss = functional.binary_cross_entropy_with_logits(
                            filler_logits[filler_valid],
                            filler_labels[filler_valid].to(dtype=logits.dtype),
                        )
                        loss = loss + self.turn_config.filler_loss_weight * filler_loss
                result.update({"loss": loss, "main_loss": main_loss, "filler_loss": filler_loss})
            return result

        def freeze_encoder(self, frozen: bool = True) -> None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(not frozen)

        def parameter_groups(
            self, *, encoder_learning_rate: float, head_learning_rate: float
        ) -> list[dict[str, Any]]:
            encoder_parameters = list(self.encoder.parameters())
            encoder_ids = {id(parameter) for parameter in encoder_parameters}
            head_parameters = [
                parameter for parameter in self.parameters() if id(parameter) not in encoder_ids
            ]
            return [
                {"params": encoder_parameters, "lr": encoder_learning_rate},
                {"params": head_parameters, "lr": head_learning_rate},
            ]

        def save_pretrained(self, output_dir: str | Path) -> None:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            metadata = {
                "turn_detector_config": self.turn_config.model_dump(mode="json"),
                "encoder_config": self.encoder.config.to_dict(),
            }
            (target / "turn_detector_config.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            try:
                from safetensors.torch import save_file

                state = {name: tensor.contiguous() for name, tensor in self.state_dict().items()}
                save_file(state, target / "model.safetensors")
            except ImportError:  # pragma: no cover - safetensors is a train dependency
                torch.save(self.state_dict(), target / "pytorch_model.bin")

        @classmethod
        def from_pretrained(cls, source: str | Path, *, map_location: str = "cpu") -> Any:
            path = Path(source)
            if not path.exists():
                try:
                    from huggingface_hub import snapshot_download
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError("Remote loading requires huggingface-hub") from exc
                path = Path(
                    snapshot_download(
                        str(source),
                        allow_patterns=[
                            "turn_detector_config.json",
                            "model.safetensors",
                            "pytorch_model.bin",
                            "policy.json",
                        ],
                    )
                )
            metadata = json.loads((path / "turn_detector_config.json").read_text())
            config = ModelConfig.model_validate(metadata["turn_detector_config"])
            model = cls(config, encoder_config=metadata["encoder_config"])
            safetensor_path = path / "model.safetensors"
            if safetensor_path.exists():
                from safetensors.torch import load_file

                state = load_file(safetensor_path, device=map_location)
            else:
                state = torch.load(
                    path / "pytorch_model.bin", map_location=map_location, weights_only=True
                )
            model.load_state_dict(state)
            return model

    return DualScaleTurnDetector


def create_turn_model(
    config: ModelConfig,
    *,
    encoder: Any | None = None,
    encoder_config: dict[str, Any] | None = None,
) -> Any:
    model_class = _build_model_class()
    return model_class(config, encoder=encoder, encoder_config=encoder_config)


def load_turn_model(source: str | Path, *, map_location: str = "cpu") -> Any:
    model_class = _build_model_class()
    return model_class.from_pretrained(source, map_location=map_location)
