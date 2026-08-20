import numpy as np

from turn_detector.config import ModelConfig, PolicyConfig
from turn_detector.inference import TurnDetector
from turn_detector.types import TurnDecision, TurnEventType, TurnPrediction


def test_stream_clock_survives_completed_turn() -> None:
    detector = TurnDetector.__new__(TurnDetector)
    detector.model_config = ModelConfig()
    detector.policy = PolicyConfig(min_silence_ms=200, timeout_ms=1_000)
    detector._reset_stream()
    prediction = TurnPrediction(0.95, TurnDecision.COMPLETE, 200, 1.0)
    detector.score = lambda *_args: prediction

    speech = np.full(1_600, 0.1, dtype=np.float32)
    silence = np.zeros(1_600, dtype=np.float32)
    started = detector.process_chunk(speech)
    assert started is not None and started.type is TurnEventType.SPEECH_STARTED
    assert detector.process_chunk(silence) is None
    completed = detector.process_chunk(silence)
    assert completed is not None and completed.timestamp_ms == 300
    next_started = detector.process_chunk(speech)
    assert next_started is not None and next_started.timestamp_ms == 400
