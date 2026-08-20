from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from turn_detector.audio import ensure_float32_mono, resample_audio, standardize_candidate_audio
from turn_detector.config import ModelConfig, PolicyConfig
from turn_detector.features import WhisperTurnFeatureExtractor
from turn_detector.types import TurnDecision, TurnEvent, TurnEventType, TurnPrediction


class TurnDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        model_config: ModelConfig | None = None,
        policy: PolicyConfig | None = None,
        intra_op_threads: int = 1,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ONNX inference requires `uv sync --extra export --extra train`"
            ) from exc
        path = Path(model_path)
        if not path.exists():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Remote loading requires huggingface-hub") from exc
            snapshot = Path(
                snapshot_download(
                    str(model_path),
                    allow_patterns=["*.onnx", "policy.json", "turn_detector_config.json"],
                )
            )
            candidates = sorted(snapshot.glob("*.int8.onnx")) or sorted(snapshot.glob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(f"No ONNX model found in {snapshot}")
            path = candidates[0]
        self.model_path = path
        self.model_config = model_config or self._load_model_config(path.parent)
        self.policy = policy or self._load_policy(path.parent)
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.feature_extractor = WhisperTurnFeatureExtractor(self.model_config)
        self._reset_stream()

    @staticmethod
    def _load_model_config(directory: Path) -> ModelConfig:
        config_path = directory / "turn_detector_config.json"
        if not config_path.exists():
            return ModelConfig()
        payload = json.loads(config_path.read_text())
        return ModelConfig.model_validate(payload.get("turn_detector_config", payload))

    @staticmethod
    def _load_policy(directory: Path) -> PolicyConfig:
        path = directory / "policy.json"
        return (
            PolicyConfig.model_validate_json(path.read_text()) if path.exists() else PolicyConfig()
        )

    @classmethod
    def from_pretrained(cls, source: str | Path, **kwargs: Any) -> TurnDetector:
        return cls(source, **kwargs)

    def score(self, audio: np.ndarray, sample_rate: int = 16_000) -> TurnPrediction:
        waveform = ensure_float32_mono(audio)
        if sample_rate != self.model_config.sample_rate:
            waveform = resample_audio(waveform, sample_rate, self.model_config.sample_rate)
        standardized = standardize_candidate_audio(
            waveform,
            self.model_config.sample_rate,
            max_seconds=self.model_config.max_seconds,
            trailing_silence_ms=self.model_config.trailing_silence_ms,
        )
        features = self.feature_extractor(standardized, return_tensors="np")
        started = time.perf_counter()
        probability = float(
            self.session.run(
                ["p_complete"],
                {
                    "input_features": np.asarray(features.input_features, dtype=np.float32),
                    "frame_mask": np.asarray(features.frame_mask, dtype=np.int64),
                },
            )[0][0, 0]
        )
        inference_ms = (time.perf_counter() - started) * 1_000
        calibrated = self._calibrate(probability)
        decision = (
            TurnDecision.COMPLETE if calibrated >= self.policy.threshold else TurnDecision.HOLD
        )
        return TurnPrediction(
            probability=calibrated,
            decision=decision,
            recommended_wait_ms=(
                self.policy.min_silence_ms
                if decision is TurnDecision.COMPLETE
                else self.policy.timeout_ms
            ),
            inference_ms=inference_ms,
        )

    def _calibrate(self, probability: float) -> float:
        clipped = float(np.clip(probability, 1e-6, 1 - 1e-6))
        logit = np.log(clipped / (1 - clipped)) / self.policy.temperature
        return float(1 / (1 + np.exp(-logit)))

    def _reset_stream(self, *, preserve_clock: bool = False) -> None:
        total_samples = getattr(self, "_total_samples", 0) if preserve_clock else 0
        self._buffer = np.zeros(0, dtype=np.float32)
        self._turn_active = False
        self._speaking = False
        self._silence_ms = 0
        self._total_samples = total_samples
        self._last_prediction: TurnPrediction | None = None
        self._scored_current_pause = False

    def process_chunk(self, pcm_chunk: np.ndarray, sample_rate: int = 16_000) -> TurnEvent | None:
        if sample_rate != self.model_config.sample_rate:
            raise ValueError(
                f"Streaming requires {self.model_config.sample_rate} Hz chunks; resample upstream"
            )
        chunk = ensure_float32_mono(pcm_chunk)
        if chunk.size == 0:
            return None
        self._total_samples += chunk.size
        max_buffer = round(60 * sample_rate)
        self._buffer = np.concatenate([self._buffer, chunk])[-max_buffer:]
        timestamp_ms = round(self._total_samples * 1_000 / sample_rate)
        chunk_ms = round(chunk.size * 1_000 / sample_rate)
        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
        is_speech = rms >= self.policy.speech_rms_threshold

        if is_speech:
            resumed = self._turn_active and not self._speaking and self._silence_ms > 0
            started = not self._turn_active
            self._turn_active = True
            self._speaking = True
            self._silence_ms = 0
            self._scored_current_pause = False
            self._last_prediction = None
            if started:
                return TurnEvent(TurnEventType.SPEECH_STARTED, timestamp_ms)
            if resumed:
                return TurnEvent(TurnEventType.SPEECH_RESUMED, timestamp_ms)
            return None

        if not self._turn_active:
            return None
        self._speaking = False
        self._silence_ms += chunk_ms
        if self._silence_ms >= self.policy.min_silence_ms and not self._scored_current_pause:
            self._last_prediction = self.score(self._buffer, sample_rate)
            self._scored_current_pause = True
            if self._last_prediction.decision is TurnDecision.COMPLETE:
                event = TurnEvent(
                    TurnEventType.TURN_COMPLETED,
                    timestamp_ms,
                    probability=self._last_prediction.probability,
                    inference_ms=self._last_prediction.inference_ms,
                )
                self._reset_stream(preserve_clock=True)
                return event
        if self._silence_ms >= self.policy.timeout_ms:
            event = TurnEvent(
                TurnEventType.TURN_COMPLETED,
                timestamp_ms,
                probability=(self._last_prediction.probability if self._last_prediction else None),
                inference_ms=(
                    self._last_prediction.inference_ms if self._last_prediction else None
                ),
            )
            self._reset_stream(preserve_clock=True)
            return event
        return None
