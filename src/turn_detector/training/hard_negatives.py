from __future__ import annotations

from pathlib import Path
from typing import Any

from turn_detector.config import ModelConfig
from turn_detector.data.records import read_manifest, write_manifest
from turn_detector.modeling import load_turn_model
from turn_detector.training.dataset import TurnAudioDataset, TurnCollator


def mine_hard_negatives(
    model_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    model_config: ModelConfig,
    batch_size: int = 64,
    top_fraction: float = 0.10,
    minimum_probability: float = 0.35,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Hard-negative mining requires the train extra") from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_turn_model(model_path, map_location=str(device)).to(device).eval()
    dataset = TurnAudioDataset(manifest_path, model_config, augment=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TurnCollator(model_config),
    )
    scores: dict[str, float] = {}
    with torch.inference_mode():
        for batch in loader:
            output = model(
                input_features=batch["input_features"].to(device),
                frame_mask=batch["frame_mask"].to(device),
            )
            for record_id, probability in zip(
                batch["ids"], output["probabilities"].float().cpu().tolist(), strict=True
            ):
                scores[record_id] = float(probability)
    candidates = [
        record
        for record in read_manifest(manifest_path)
        if not record.endpoint_bool and scores.get(record.id, 0.0) >= minimum_probability
    ]
    candidates.sort(key=lambda record: scores[record.id], reverse=True)
    keep = max(1, round(len(dataset.records) * top_fraction)) if candidates else 0
    selected = [
        record.model_copy(
            update={"is_hard_negative": True, "sampling_weight": record.sampling_weight * 3.0}
        )
        for record in candidates[:keep]
    ]
    write_manifest(selected, output_path)
    return {
        "scored": len(dataset.records),
        "eligible": len(candidates),
        "selected": len(selected),
        "minimum_probability": minimum_probability,
        "top_probability": scores[selected[0].id] if selected else None,
    }
