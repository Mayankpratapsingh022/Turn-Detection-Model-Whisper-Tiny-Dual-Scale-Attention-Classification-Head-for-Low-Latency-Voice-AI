from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from turn_detector.audio import (
    analyze_quality,
    find_internal_pauses,
    load_audio,
    resample_audio,
    save_audio,
    speech_segments,
    standardize_candidate_audio,
)
from turn_detector.config import DataConfig
from turn_detector.data.duplicates import acoustic_fingerprint, duplicate_group, waveform_hash
from turn_detector.data.records import AudioRecord, read_manifest, write_json, write_manifest
from turn_detector.data.sampling import category_weight
from turn_detector.data.splits import assert_no_group_leakage, assign_grouped_stratified_splits

SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_id(value: str) -> str:
    cleaned = SAFE_ID.sub("_", value).strip("._")
    if cleaned:
        return cleaned[:120]
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def iter_huggingface_rows(
    repo_id: str,
    split: str = "train",
    *,
    revision: str | None = None,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Dataset preparation requires `uv sync --extra data`") from exc

    dataset = load_dataset(repo_id, split=split, streaming=True, revision=revision)
    with suppress(AttributeError, TypeError, ValueError):
        dataset = dataset.cast_column("audio", Audio(decode=False))
        # Older/newer datasets releases can already expose undecoded mappings.
    yield from dataset


def _relative_audio_path(output_dir: Path, repo_id: str, record_id: str) -> Path:
    repo_slug = repo_id.replace("/", "__")
    return Path("audio") / repo_slug / f"{_safe_id(record_id)}.flac"


def _record_from_audio(
    *,
    row: dict[str, Any],
    record_id: str,
    parent_id: str,
    audio: np.ndarray,
    original_audio: np.ndarray,
    config: DataConfig,
    source_repo: str,
    audio_path: Path,
    example_kind: Literal["original", "causal_internal_pause"],
    endpoint_bool: bool,
    pause_duration_ms: int | None,
    valid_samples: int,
    parent_group: str | None = None,
) -> AudioRecord:
    quality = analyze_quality(original_audio, config.sample_rate)
    audio_digest = waveform_hash(audio)
    fingerprint = acoustic_fingerprint(audio, config.sample_rate)
    group = duplicate_group(audio_digest, fingerprint, parent_group=parent_group)
    speech_ratio = quality.speech_seconds / max(quality.duration_seconds, 1e-6)
    record = AudioRecord(
        id=record_id,
        parent_id=parent_id,
        audio_path=str(audio_path),
        source_repo=source_repo,
        source_dataset=str(row.get("dataset") or "unknown"),
        language=cast(Literal["hin", "eng"], str(row["language"])),
        endpoint_bool=endpoint_bool,
        midfiller=_optional_bool(row.get("midfiller")),
        endfiller=_optional_bool(row.get("endfiller")),
        synthetic=_optional_bool(row.get("synthetic")),
        example_kind=example_kind,
        split="unassigned",
        duration_seconds=quality.duration_seconds,
        valid_samples=valid_samples,
        speech_seconds=quality.speech_seconds,
        speech_ratio=speech_ratio,
        peak_dbfs=quality.peak_dbfs,
        rms_dbfs=quality.rms_dbfs,
        clipping_ratio=quality.clipping_ratio,
        silence_ratio=quality.silence_ratio,
        pause_duration_ms=pause_duration_ms,
        audio_hash=audio_digest,
        acoustic_fingerprint=fingerprint,
        duplicate_group=group,
        quality_status=quality.reason or "ok",
    )
    return record.model_copy(update={"sampling_weight": category_weight(record)})


def _prepare_row(
    row: dict[str, Any],
    *,
    config: DataConfig,
    source_repo: str,
    output_dir: Path,
    forced_split: str | None,
) -> tuple[list[AudioRecord], dict[str, Any]]:
    language_value = str(row.get("language") or "")
    if language_value not in config.languages:
        return [], {"status": "excluded_language", "language": language_value}
    language = cast(Literal["hin", "eng"], language_value)
    record_id = str(row.get("id") or hashlib.sha256(repr(row).encode()).hexdigest()[:24])
    try:
        waveform, sample_rate = load_audio(row["audio"])
        waveform = resample_audio(waveform, sample_rate, config.sample_rate)
    except Exception as exc:
        return [], {"status": "decode_error", "id": record_id, "error": repr(exc)}

    quality = analyze_quality(waveform, config.sample_rate)
    if not quality.valid:
        return [], {
            "status": quality.reason or "invalid",
            "id": record_id,
            "language": language,
            **asdict(quality),
        }

    standardized = standardize_candidate_audio(
        waveform,
        config.sample_rate,
        max_seconds=config.max_seconds,
        trailing_silence_ms=config.trailing_silence_ms,
    )
    relative_path = _relative_audio_path(output_dir, source_repo, record_id)
    if config.cache_audio:
        save_audio(output_dir / relative_path, standardized.waveform, config.sample_rate)

    segments = speech_segments(waveform, config.sample_rate)
    final_pause_ms = None
    if segments:
        final_pause_ms = round((len(waveform) - segments[-1][1]) * 1_000 / config.sample_rate)
    original = _record_from_audio(
        row=row,
        record_id=record_id,
        parent_id=record_id,
        audio=standardized.waveform,
        original_audio=waveform,
        config=config,
        source_repo=source_repo,
        audio_path=relative_path,
        example_kind="original",
        endpoint_bool=bool(row.get("endpoint_bool")),
        pause_duration_ms=final_pause_ms,
        valid_samples=standardized.valid_samples,
    )
    if forced_split:
        original = original.model_copy(update={"split": forced_split})
    records = [original]

    pauses = find_internal_pauses(
        waveform,
        config.sample_rate,
        min_pause_ms=config.min_internal_pause_ms,
        max_pause_ms=config.max_internal_pause_ms,
        min_future_speech_ms=config.min_future_speech_ms,
    )[: config.max_internal_pauses_per_clip]
    decision_offset = round(config.trailing_silence_ms * config.sample_rate / 1_000)
    for pause_index, pause in enumerate(pauses):
        causal_id = f"{record_id}__pause_{pause_index:02d}"
        causal_source = waveform[: min(len(waveform), pause.start_sample + decision_offset)]
        causal_standardized = standardize_candidate_audio(
            causal_source,
            config.sample_rate,
            max_seconds=config.max_seconds,
            trailing_silence_ms=config.trailing_silence_ms,
        )
        causal_path = _relative_audio_path(output_dir, source_repo, causal_id)
        if config.cache_audio:
            save_audio(output_dir / causal_path, causal_standardized.waveform, config.sample_rate)
        causal_record = _record_from_audio(
            row=row,
            record_id=causal_id,
            parent_id=record_id,
            audio=causal_standardized.waveform,
            original_audio=causal_source,
            config=config,
            source_repo=source_repo,
            audio_path=causal_path,
            example_kind="causal_internal_pause",
            endpoint_bool=False,
            pause_duration_ms=pause.duration_ms,
            valid_samples=causal_standardized.valid_samples,
            parent_group=original.duplicate_group,
        )
        if forced_split:
            causal_record = causal_record.model_copy(update={"split": forced_split})
        records.append(causal_record)

    return records, {
        "status": "ok",
        "id": record_id,
        "language": language,
        "derived_pauses": len(records) - 1,
    }


def summarize_records(records: Iterable[AudioRecord]) -> dict[str, Any]:
    materialized = list(records)
    counts: dict[str, Counter[Any]] = {
        "language": Counter(),
        "label": Counter(),
        "split": Counter(),
        "source_dataset": Counter(),
        "speech_mix": Counter(),
        "example_kind": Counter(),
        "quality_status": Counter(),
    }
    for record in materialized:
        counts["language"][record.language] += 1
        counts["label"]["eot" if record.endpoint_bool else "hold"] += 1
        counts["split"][record.split] += 1
        counts["source_dataset"][record.source_dataset] += 1
        counts["speech_mix"][record.speech_mix] += 1
        counts["example_kind"][record.example_kind] += 1
        counts["quality_status"][record.quality_status] += 1
    return {
        "num_records": len(materialized),
        "num_parent_utterances": len({record.parent_id for record in materialized}),
        "num_duplicate_groups": len({record.duplicate_group for record in materialized}),
        **{name: dict(counter) for name, counter in counts.items()},
    }


def prepare_dataset(
    config: DataConfig,
    *,
    repo_id: str | None = None,
    forced_split: str | None = None,
    rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_repo = repo_id or config.train_repo
    source_revision = (
        config.test_revision if source_repo == config.test_repo else config.train_revision
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = (
        rows
        if rows is not None
        else iter_huggingface_rows(source_repo, config.split, revision=source_revision)
    )

    records: list[AudioRecord] = []
    audit_counts: Counter[str] = Counter()
    audit_failures: list[dict[str, Any]] = []
    accepted_parents = 0
    for row in source_rows:
        if config.limit is not None and accepted_parents >= config.limit:
            break
        prepared, audit = _prepare_row(
            row,
            config=config,
            source_repo=source_repo,
            output_dir=output_dir,
            forced_split=forced_split,
        )
        audit_counts[str(audit["status"])] += 1
        if audit["status"] not in {"ok", "excluded_language"} and len(audit_failures) < 1_000:
            audit_failures.append(audit)
        if prepared:
            accepted_parents += 1
            records.extend(prepared)

    if forced_split is None:
        records = assign_grouped_stratified_splits(
            records,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )
    assert_no_group_leakage(records)

    manifests: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        subset = [record for record in records if record.split == split_name]
        if subset and config.cache_audio:
            manifest_path = output_dir / f"{split_name}.jsonl"
            write_manifest(subset, manifest_path)
            manifests[split_name] = str(manifest_path)

    summary = {
        "source_repo": source_repo,
        "source_revision": source_revision or "main (unpinned)",
        "languages": list(config.languages),
        "accepted_parent_utterances": accepted_parents,
        "audit_counts": dict(audit_counts),
        "audit_failures": audit_failures,
        "manifests": manifests,
        "records": summarize_records(records),
        "config": config.model_dump(mode="json"),
    }
    slug = source_repo.replace("/", "__")
    write_json(summary, output_dir / f"audit__{slug}.json")
    return summary


def remove_cross_corpus_duplicates(
    reference: list[AudioRecord], candidate: list[AudioRecord]
) -> tuple[list[AudioRecord], dict[str, int]]:
    """Drop complete candidate parent groups that overlap a reference corpus."""

    reference_hashes = {record.audio_hash for record in reference}
    reference_groups = {record.duplicate_group for record in reference}
    blocked_parents = {
        record.parent_id
        for record in candidate
        if record.audio_hash in reference_hashes or record.duplicate_group in reference_groups
    }
    filtered = [record for record in candidate if record.parent_id not in blocked_parents]
    return filtered, {
        "removed_parent_utterances": len(blocked_parents),
        "removed_records": len(candidate) - len(filtered),
        "remaining_records": len(filtered),
    }


def prepare_train_and_test(config: DataConfig) -> dict[str, Any]:
    train_summary = prepare_dataset(config, repo_id=config.train_repo)
    test_summary = prepare_dataset(config, repo_id=config.test_repo, forced_split="test")
    if config.cache_audio:
        reference = [
            record
            for name in ("train.jsonl", "validation.jsonl")
            if (path := config.output_dir / name).exists()
            for record in read_manifest(path)
        ]
        test_manifest = config.output_dir / "test.jsonl"
        if test_manifest.exists():
            test_records = list(read_manifest(test_manifest))
            test_records, cross_split = remove_cross_corpus_duplicates(reference, test_records)
            write_manifest(test_records, test_manifest)
            test_summary["cross_corpus_dedup"] = cross_split
            test_summary["records"] = summarize_records(test_records)
            slug = config.test_repo.replace("/", "__")
            write_json(test_summary, config.output_dir / f"audit__{slug}.json")
    return {"train": train_summary, "test": test_summary}


def save_summary_markdown(summary: dict[str, Any], path: str | Path) -> None:
    lines = ["# Dataset audit", "", "```json", json.dumps(summary, indent=2), "```", ""]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
