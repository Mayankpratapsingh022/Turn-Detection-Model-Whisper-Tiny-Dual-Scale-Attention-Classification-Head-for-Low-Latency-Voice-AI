from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AudioRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    parent_id: str
    audio_path: str
    source_repo: str
    source_dataset: str = "unknown"
    language: Literal["hin", "eng"]
    endpoint_bool: bool
    midfiller: bool | None = None
    endfiller: bool | None = None
    synthetic: bool | None = None
    example_kind: Literal["original", "causal_internal_pause"] = "original"
    split: Literal["train", "validation", "test", "unassigned"] = "unassigned"
    duration_seconds: float
    valid_samples: int
    speech_seconds: float
    speech_ratio: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    pause_duration_ms: int | None = None
    audio_hash: str
    acoustic_fingerprint: str
    duplicate_group: str
    quality_status: str = "ok"
    asr_text: str | None = None
    asr_confidence: float | None = None
    speech_mix: Literal["hinglish_high_confidence", "hindi", "english", "uncertain", "untagged"] = (
        "untagged"
    )
    is_hard_negative: bool = False
    sampling_weight: float = Field(default=1.0, gt=0)

    @property
    def label(self) -> int:
        return int(self.endpoint_bool)

    @property
    def filler_present(self) -> bool | None:
        if self.midfiller is None and self.endfiller is None:
            return None
        return bool(self.midfiller or self.endfiller)

    def resolved_audio_path(self, manifest_path: str | Path) -> Path:
        path = Path(self.audio_path)
        if path.is_absolute():
            return path
        return Path(manifest_path).parent / path


def read_manifest(path: str | Path) -> Iterator[AudioRecord]:
    source = Path(path)
    if source.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Reading Parquet requires the 'data' extra") from exc
        table = pq.read_table(source)
        for row in table.to_pylist():
            yield AudioRecord.model_validate(row)
        return
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield AudioRecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"Invalid manifest row {line_number} in {source}") from exc


def write_manifest(records: Iterable[AudioRecord], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(records)
    if target.suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Writing Parquet requires the 'data' extra") from exc
        table = pa.Table.from_pylist([record.model_dump(mode="json") for record in materialized])
        pq.write_table(table, target, compression="zstd")
        return len(materialized)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in materialized:
                handle.write(record.model_dump_json())
                handle.write("\n")
        os.replace(temporary_name, target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return len(materialized)


def write_json(payload: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
