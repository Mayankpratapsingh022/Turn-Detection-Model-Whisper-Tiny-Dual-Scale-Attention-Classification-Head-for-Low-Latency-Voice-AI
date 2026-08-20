import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.ml


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="torch/transformers not installed",
)
def test_tiny_random_model_forward() -> None:
    import torch
    from transformers import WhisperConfig

    from turn_detector.config import ModelConfig
    from turn_detector.modeling import create_turn_model

    config = ModelConfig(
        mel_frames=8,
        max_seconds=0.08,
        tail_seconds=0.04,
        attention_hidden_size=8,
        classifier_hidden_size=16,
        classifier_bottleneck_size=8,
    )
    encoder_config = WhisperConfig(
        num_mel_bins=80,
        d_model=32,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_layers=1,
        decoder_attention_heads=4,
        decoder_ffn_dim=64,
        max_source_positions=4,
    ).to_dict()
    model = create_turn_model(config, encoder_config=encoder_config)
    output = model(
        input_features=torch.randn(2, 80, 8),
        frame_mask=torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [1] * 8]),
        labels=torch.tensor([0, 1]),
        filler_labels=torch.tensor([[-1, -1], [1, 0]]),
    )
    assert output["probabilities"].shape == (2,)
    assert output["filler_logits"].shape == (2, 2)
    assert output["loss"].ndim == 0


@pytest.mark.skipif(
    any(importlib.util.find_spec(package) is None for package in ("torch", "transformers", "onnx")),
    reason="train/export dependencies not installed",
)
def test_tiny_model_save_load_and_onnx_export(tmp_path: Path) -> None:
    from transformers import WhisperConfig

    from turn_detector.config import ModelConfig
    from turn_detector.data.records import AudioRecord, write_manifest
    from turn_detector.export import export_onnx
    from turn_detector.modeling import create_turn_model, load_turn_model

    config = ModelConfig(
        mel_frames=100,
        max_seconds=1.0,
        tail_seconds=0.4,
        attention_hidden_size=8,
        classifier_hidden_size=16,
        classifier_bottleneck_size=8,
    )
    encoder_config = WhisperConfig(
        num_mel_bins=80,
        d_model=32,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_layers=1,
        decoder_attention_heads=4,
        decoder_ffn_dim=64,
        max_source_positions=50,
    ).to_dict()
    checkpoint = tmp_path / "checkpoint"
    create_turn_model(config, encoder_config=encoder_config).save_pretrained(checkpoint)
    assert load_turn_model(checkpoint).turn_config == config
    waveform = (0.2 * np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000)).astype(np.float32)
    sf.write(tmp_path / "sample.flac", waveform, 16_000)
    manifest = tmp_path / "validation.jsonl"
    write_manifest(
        [
            AudioRecord(
                id="sample",
                parent_id="sample",
                audio_path="sample.flac",
                source_repo="fixture/repo",
                language="hin",
                endpoint_bool=True,
                duration_seconds=1.0,
                valid_samples=16_000,
                speech_seconds=1.0,
                speech_ratio=1.0,
                peak_dbfs=-3.0,
                rms_dbfs=-12.0,
                clipping_ratio=0.0,
                silence_ratio=0.0,
                audio_hash="sample",
                acoustic_fingerprint="sample",
                duplicate_group="sample",
            )
        ],
        manifest,
    )
    report = export_onnx(
        checkpoint,
        tmp_path / "model.onnx",
        quantize=True,
        calibration_manifest=manifest,
        calibration_samples=1,
    )
    assert report["fp32_max_probability_difference"] < 0.01
    assert report["quantization_method"] == "static_qdq_entropy"
    assert (tmp_path / "model.int8.onnx").exists()
    assert (tmp_path / "turn_detector_config.json").exists()
