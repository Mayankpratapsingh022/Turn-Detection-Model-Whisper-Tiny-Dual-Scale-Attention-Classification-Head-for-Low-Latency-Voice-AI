from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

DATASET_REPO = "pipecat-ai/smart-turn-data-v3.2-train"
DATASET_REVISION = "e564e2ac567f774d1880aa1db6ce97afb8c519b7"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
MAX_AUDIO_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class DemoExample:
    path: Path
    label: str


@dataclass(frozen=True)
class _ExampleSpec:
    row_index: int
    language: str
    endpoint: bool
    label: str


# These rows are deterministic cases from the pinned public training snapshot. Labels are
# deliberately visible so the presets explain the task instead of pretending to be a blind test.
_EXAMPLE_SPECS = (
    _ExampleSpec(33, "hin", False, "Hindi · mid-filler sample 1"),
    _ExampleSpec(43, "hin", False, "Hindi · mid-filler sample 2"),
    _ExampleSpec(7, "hin", True, "Hindi · filler speech sample"),
    _ExampleSpec(79, "hin", False, "Hindi · end-filler sample"),
    _ExampleSpec(74, "hin", True, "Hindi · speech sample"),
    _ExampleSpec(30, "eng", False, "English · end-filler sample"),
    _ExampleSpec(76, "eng", True, "English · mid-filler sample"),
    _ExampleSpec(4, "eng", False, "English · natural pause sample"),
    _ExampleSpec(1, "eng", True, "English · natural speech sample"),
)


def _request_bytes(url: str, *, timeout_seconds: float, maximum_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "hinglish-turn-detector-demo/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        declared_size = response.headers.get("Content-Length")
        if declared_size is not None and int(declared_size) > maximum_bytes:
            raise ValueError(f"Remote asset is larger than {maximum_bytes} bytes.")
        payload = response.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"Remote asset is larger than {maximum_bytes} bytes.")
    return payload


def _first_rows_url() -> str:
    dataset = quote(DATASET_REPO, safe="")
    return (
        "https://datasets-server.huggingface.co/first-rows"
        f"?dataset={dataset}&config={DATASET_CONFIG}&split={DATASET_SPLIT}"
    )


def _validate_audio_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "datasets-server.huggingface.co":
        raise ValueError("Dataset viewer returned an unexpected audio host.")
    if f"/{DATASET_REVISION}/" not in parsed.path:
        raise ValueError("Dataset viewer is not serving the pinned dataset revision.")


def _source_url(row: dict[str, Any]) -> str:
    audio = row.get("audio")
    if not isinstance(audio, list) or not audio or not isinstance(audio[0], dict):
        raise ValueError("Dataset row does not contain a downloadable audio asset.")
    source = audio[0].get("src")
    if not isinstance(source, str):
        raise ValueError("Dataset row audio URL is missing.")
    _validate_audio_url(source)
    return source


def load_demo_examples(
    cache_dir: str | Path | None = None,
    *,
    timeout_seconds: float = 15.0,
) -> list[DemoExample]:
    """Fetch a few pinned public examples into an ephemeral runtime cache.

    This function is intentionally fail-soft. A network or upstream viewer failure returns an
    empty list so the microphone, upload control, and prediction API remain available.
    """

    try:
        payload = _request_bytes(
            _first_rows_url(),
            timeout_seconds=timeout_seconds,
            maximum_bytes=4 * 1024 * 1024,
        )
        document = json.loads(payload)
        if document.get("dataset") != DATASET_REPO:
            raise ValueError("Dataset viewer returned a different repository.")
        rows = {
            int(item["row_idx"]): item["row"]
            for item in document.get("rows", [])
            if isinstance(item, dict) and isinstance(item.get("row"), dict)
        }
        root = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(tempfile.gettempdir()) / "hinglish-turn-detector-demo" / DATASET_REVISION[:12]
        )
        root.mkdir(parents=True, exist_ok=True)

        examples: list[DemoExample] = []
        for spec in _EXAMPLE_SPECS:
            row = rows.get(spec.row_index)
            if row is None:
                raise ValueError(f"Pinned example row {spec.row_index} is unavailable.")
            if (
                row.get("language") != spec.language
                or row.get("endpoint_bool") is not spec.endpoint
            ):
                raise ValueError(f"Pinned example row {spec.row_index} metadata changed.")

            target = root / f"row-{spec.row_index}-{spec.language}.wav"
            if not target.is_file() or target.stat().st_size < 44:
                audio = _request_bytes(
                    _source_url(row),
                    timeout_seconds=timeout_seconds,
                    maximum_bytes=MAX_AUDIO_BYTES,
                )
                temporary = target.with_suffix(".wav.part")
                temporary.write_bytes(audio)
                temporary.replace(target)
            examples.append(DemoExample(path=target, label=spec.label))
        return examples
    except (OSError, TimeoutError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return []
