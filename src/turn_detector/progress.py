from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def _progress_enabled() -> bool:
    value = os.getenv("TURN_DETECTOR_PROGRESS", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def progress_bar(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    description: str,
    unit: str,
    leave: bool = True,
) -> Any:
    """Create a consistent progress bar that remains visible in tmux/tee logs."""
    return tqdm(
        iterable,
        total=total,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
        mininterval=1.0,
        maxinterval=10.0,
        smoothing=0.1,
        leave=leave,
        disable=not _progress_enabled(),
    )


def log_event(stage: str, event: str, **details: Any) -> None:
    """Emit a timestamped, grep-friendly lifecycle event without logging secrets."""
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    suffix = " ".join(
        f"{key}={value}" for key, value in details.items() if value is not None and value != ""
    )
    message = f"[{timestamp}] [{stage}] {event}"
    if suffix:
        message = f"{message} | {suffix}"
    tqdm.write(message)
