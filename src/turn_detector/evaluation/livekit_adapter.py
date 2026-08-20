from __future__ import annotations

import os

import numpy as np

from turn_detector.audio import resample_audio, standardize_candidate_audio
from turn_detector.inference import TurnDetector


class HinglishTurnAudioAdapter:
    """Duck-typed pointwise adapter for https://github.com/livekit/eot-bench."""

    adapter_id = "hinglish-turn-8m"
    display_name = "HinglishTurn-8M"
    score_point = 0.2

    def __init__(self, model_path: str | None = None) -> None:
        resolved = model_path or os.environ.get(
            "HINGLISH_TURN_MODEL", "artifacts/export/hinglish-turn.int8.onnx"
        )
        self.detector = TurnDetector(resolved)

    def supports_language(self, lang_code: str) -> bool:
        return lang_code.lower() in {"hi", "hin", "en", "eng", "hi-in", "en-in"}

    def predict_batch(self, batch: list[dict[str, object]]) -> list[float]:
        probabilities: list[float] = []
        for item in batch:
            audio = item.get("audio")
            if not isinstance(audio, dict):
                raise ValueError("Adapter batch items must include decoded audio mappings")
            waveform = np.asarray(audio["array"], dtype=np.float32)
            sample_rate = int(audio["sampling_rate"])
            if sample_rate != self.detector.model_config.sample_rate:
                waveform = resample_audio(
                    waveform, sample_rate, self.detector.model_config.sample_rate
                )
                sample_rate = self.detector.model_config.sample_rate
            # Keep the score semantic: normalize any silence span to the same 200 ms.
            standardized = standardize_candidate_audio(
                waveform,
                sample_rate,
                max_seconds=self.detector.model_config.max_seconds,
                trailing_silence_ms=self.detector.model_config.trailing_silence_ms,
            )
            probabilities.append(
                self.detector.score(standardized.waveform, sample_rate).probability
            )
        return probabilities
