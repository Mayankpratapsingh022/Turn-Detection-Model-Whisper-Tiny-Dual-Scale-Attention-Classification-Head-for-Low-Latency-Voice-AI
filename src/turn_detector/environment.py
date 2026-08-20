from __future__ import annotations

from pathlib import Path


def load_project_env(path: str | Path = ".env", *, override: bool = False) -> bool:
    """Load local secrets without ever serializing or returning their values."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - installed with the core package
        raise RuntimeError("Environment loading requires `python-dotenv`") from exc
    return bool(load_dotenv(dotenv_path=Path(path), override=override))
