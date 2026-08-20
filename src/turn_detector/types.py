from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class TurnDecision(StrEnum):
    COMPLETE = "complete"
    HOLD = "hold"


class TurnEventType(StrEnum):
    SPEECH_STARTED = "speech_started"
    SPEECH_RESUMED = "speech_resumed"
    TURN_COMPLETED = "turn_completed"


@dataclass(frozen=True, slots=True)
class TurnPrediction:
    probability: float
    decision: TurnDecision
    recommended_wait_ms: int
    inference_ms: float

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decision"] = self.decision.value
        return value


@dataclass(frozen=True, slots=True)
class TurnEvent:
    type: TurnEventType
    timestamp_ms: int
    probability: float | None = None
    inference_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["type"] = self.type.value
        return value
