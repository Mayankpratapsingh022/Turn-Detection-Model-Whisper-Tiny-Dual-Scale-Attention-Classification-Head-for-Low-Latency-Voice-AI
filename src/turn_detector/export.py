from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from turn_detector.config import PolicyConfig
from turn_detector.features import WhisperTurnFeatureExtractor
from turn_detector.modeling import load_turn_model
from turn_detector.training.dataset import TurnAudioDataset


def _calibration_reader(
    manifest_path: str | Path,
    model_config: Any,
    *,
    max_samples: int,
) -> Any:
    try:
        from onnxruntime.quantization import CalibrationDataReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Static quantization requires the export extra") from exc

    dataset = TurnAudioDataset(manifest_path, model_config, augment=False)
    if not dataset.records:
        raise ValueError("Static quantization calibration manifest is empty")
    extractor = WhisperTurnFeatureExtractor(model_config)

    class ManifestCalibrationReader(CalibrationDataReader):
        def __init__(self) -> None:
            self.indices = list(range(min(max_samples, len(dataset))))
            self.position = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self.position >= len(self.indices):
                return None
            example = dataset[self.indices[self.position]]
            self.position += 1
            features = extractor(example["audio"], return_tensors="np")
            return {
                "input_features": np.asarray(features.input_features, dtype=np.float32),
                "frame_mask": np.asarray(features.frame_mask, dtype=np.int64),
            }

        def rewind(self) -> None:
            self.position = 0

    return ManifestCalibrationReader()


def export_onnx(
    model_path: str | Path,
    output_path: str | Path,
    *,
    policy: PolicyConfig | None = None,
    opset: int = 18,
    quantize: bool = True,
    calibration_manifest: str | Path | None = None,
    calibration_samples: int = 1_024,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ONNX export requires the train extra") from exc

    model = load_turn_model(model_path).cpu().eval()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_features: Any, frame_mask: Any) -> Any:
            return self.inner(
                input_features=input_features,
                frame_mask=frame_mask,
            )["probabilities"].unsqueeze(-1)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(model)
    dummy_features = torch.randn(2, model.turn_config.n_mels, model.turn_config.mel_frames)
    dummy_mask = torch.ones(2, model.turn_config.mel_frames, dtype=torch.int64)
    batch_one_features = torch.randn(1, model.turn_config.n_mels, model.turn_config.mel_frames)
    batch_one_mask = torch.ones(1, model.turn_config.mel_frames, dtype=torch.int64)
    batch_one_mask[:, : model.turn_config.mel_frames // 2] = 0
    with torch.inference_mode():
        expected = wrapper(dummy_features, dummy_mask).numpy()
        batch_one_expected = wrapper(batch_one_features, batch_one_mask).numpy()
    torch.onnx.export(
        wrapper,
        (dummy_features, dummy_mask),
        target,
        input_names=["input_features", "frame_mask"],
        output_names=["p_complete"],
        dynamic_axes={
            "input_features": {0: "batch"},
            "frame_mask": {0: "batch"},
            "p_complete": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ONNX verification requires the export extra") from exc
    onnx.checker.check_model(onnx.load(target))
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["p_complete"],
        {
            "input_features": dummy_features.numpy(),
            "frame_mask": dummy_mask.numpy(),
        },
    )[0]
    batch_one_actual = session.run(
        ["p_complete"],
        {
            "input_features": batch_one_features.numpy(),
            "frame_mask": batch_one_mask.numpy(),
        },
    )[0]
    fp32_max_difference = float(
        max(
            np.max(np.abs(expected - actual)),
            np.max(np.abs(batch_one_expected - batch_one_actual)),
        )
    )
    if fp32_max_difference >= 0.01:
        raise RuntimeError(f"PyTorch/ONNX parity failed: max difference {fp32_max_difference}")

    quantized_path: Path | None = None
    int8_max_difference: float | None = None
    int8_mean_difference: float | None = None
    quantization_method: str | None = None
    if quantize:
        quantized_path = target.with_name(f"{target.stem}.int8{target.suffix}")
        parity_inputs: list[dict[str, np.ndarray]] = []
        if calibration_manifest is not None:
            from onnxruntime.quantization import (
                CalibrationMethod,
                QuantFormat,
                QuantType,
                quant_pre_process,
                quantize_static,
            )

            calibration_path = Path(calibration_manifest)
            if not calibration_path.exists():
                raise FileNotFoundError(f"Calibration manifest not found: {calibration_path}")
            reader = _calibration_reader(
                calibration_path,
                model.turn_config,
                max_samples=calibration_samples,
            )
            preprocessed = target.with_name(f".{target.stem}.preprocessed{target.suffix}")
            try:
                quant_pre_process(
                    str(target),
                    str(preprocessed),
                    skip_optimization=False,
                    skip_symbolic_shape=True,
                    verbose=0,
                )
                quantize_static(
                    model_input=str(preprocessed),
                    model_output=str(quantized_path),
                    calibration_data_reader=reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    calibrate_method=CalibrationMethod.Entropy,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                )
            finally:
                preprocessed.unlink(missing_ok=True)
            reader.rewind()
            for _ in range(min(32, calibration_samples)):
                batch = reader.get_next()
                if batch is None:
                    break
                parity_inputs.append(batch)
            quantization_method = "static_qdq_entropy"
        else:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                model_input=target,
                model_output=quantized_path,
                weight_type=QuantType.QInt8,
                per_channel=True,
                op_types_to_quantize=["MatMul", "Gemm"],
            )
            parity_inputs.append(
                {
                    "input_features": dummy_features.numpy(),
                    "frame_mask": dummy_mask.numpy(),
                }
            )
            quantization_method = "dynamic_weight_only"
        quantized_session = ort.InferenceSession(
            str(quantized_path), providers=["CPUExecutionProvider"]
        )
        differences: list[float] = []
        for inputs in parity_inputs:
            fp32_output = session.run(["p_complete"], inputs)[0]
            quantized_output = quantized_session.run(["p_complete"], inputs)[0]
            differences.extend(np.abs(fp32_output - quantized_output).reshape(-1).tolist())
        if differences:
            int8_max_difference = float(np.max(differences))
            int8_mean_difference = float(np.mean(differences))

    metadata = {
        "turn_detector_config": model.turn_config.model_dump(mode="json"),
        "encoder_config": model.encoder.config.to_dict(),
    }
    (target.parent / "turn_detector_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    policy_value = policy or PolicyConfig()
    (target.parent / "policy.json").write_text(
        json.dumps(policy_value.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    report = {
        "fp32_path": str(target),
        "fp32_size_bytes": target.stat().st_size,
        "fp32_max_probability_difference": fp32_max_difference,
        "int8_path": str(quantized_path) if quantized_path else None,
        "int8_size_bytes": quantized_path.stat().st_size if quantized_path else None,
        "int8_max_probability_difference": int8_max_difference,
        "int8_mean_probability_difference": int8_mean_difference,
        "meets_int8_probability_parity_target": (
            int8_max_difference < 0.05 if int8_max_difference is not None else None
        ),
        "quantization_method": quantization_method,
        "calibration_manifest": str(calibration_manifest) if calibration_manifest else None,
        "calibration_samples": calibration_samples if calibration_manifest else 0,
        "meets_10mb_target": (
            quantized_path.stat().st_size <= 10 * 1024 * 1024 if quantized_path else None
        ),
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "verified_batch_sizes": [1, 2],
        "opset": opset,
    }
    (target.parent / "export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
