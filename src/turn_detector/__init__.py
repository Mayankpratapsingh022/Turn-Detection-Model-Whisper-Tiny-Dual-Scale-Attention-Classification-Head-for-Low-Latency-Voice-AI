"""Audio-native Hindi/Hinglish end-of-turn detection."""

from typing import TYPE_CHECKING, Any

from turn_detector.types import TurnDecision, TurnEvent, TurnPrediction

if TYPE_CHECKING:
    from turn_detector.config import AppConfig

__all__ = ["AppConfig", "TurnDecision", "TurnEvent", "TurnPrediction"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "AppConfig":
        from turn_detector.config import AppConfig

        return AppConfig
    raise AttributeError(name)
