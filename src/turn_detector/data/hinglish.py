from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from turn_detector.audio import load_audio
from turn_detector.data.records import AudioRecord, read_manifest, write_manifest
from turn_detector.data.sampling import category_weight

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

ROMAN_HINDI = {
    "acha",
    "accha",
    "abhi",
    "aap",
    "aur",
    "bas",
    "bhai",
    "bol",
    "chahiye",
    "hai",
    "hain",
    "haan",
    "hum",
    "kal",
    "kar",
    "karna",
    "kya",
    "lekin",
    "main",
    "matlab",
    "mera",
    "meri",
    "mujhe",
    "nahi",
    "nhi",
    "par",
    "raha",
    "rahi",
    "tha",
    "thi",
    "theek",
    "toh",
    "tum",
    "wala",
    "wali",
    "yaar",
    "yeh",
}

ENGLISH_CONTENT = {
    "account",
    "actually",
    "address",
    "app",
    "appointment",
    "booking",
    "cab",
    "cancel",
    "card",
    "delivery",
    "email",
    "flight",
    "issue",
    "leave",
    "meeting",
    "number",
    "office",
    "order",
    "payment",
    "please",
    "refund",
    "return",
    "support",
    "ticket",
    "tomorrow",
    "update",
}


def classify_speech_mix(
    text: str,
    source_language: str,
    *,
    asr_confidence: float | None = None,
) -> str:
    normalized = text.strip().lower()
    if not normalized or (asr_confidence is not None and asr_confidence < 0.35):
        return "uncertain"
    devanagari_tokens = DEVANAGARI_RE.findall(normalized)
    latin_tokens = [token.lower() for token in LATIN_RE.findall(normalized)]
    roman_hindi_count = sum(token in ROMAN_HINDI for token in latin_tokens)
    english_count = sum(token in ENGLISH_CONTENT for token in latin_tokens)

    has_hindi = len(devanagari_tokens) >= 2 or roman_hindi_count >= 2
    has_english = english_count >= 1
    if has_hindi and has_english:
        return "hinglish_high_confidence"
    if len(devanagari_tokens) >= 2 or roman_hindi_count >= 2 or source_language == "hin":
        return "hindi"
    if len(latin_tokens) >= 2 or source_language == "eng":
        return "english"
    return "uncertain"


def _resumable_records(
    manifest_path: str | Path,
    output_path: str | Path,
) -> list[AudioRecord]:
    source = Path(manifest_path)
    output = Path(output_path)
    if source.resolve().parent != output.resolve().parent:
        raise ValueError(
            "Tagged manifests must stay beside the source so relative audio paths work"
        )
    records = list(read_manifest(source))
    if not output.exists():
        return records
    existing = list(read_manifest(output))
    if [record.id for record in existing] != [record.id for record in records]:
        raise ValueError("Existing tagged manifest does not align with the source manifest")
    return existing


def _load_asr_model(model_name: str, device: str, compute_type: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ASR tagging requires `uv sync --extra data`") from exc
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _tag_with_model(
    model: Any,
    manifest_path: str | Path,
    output_path: str | Path,
    records: list[AudioRecord],
    *,
    limit: int | None,
    checkpoint_every: int,
) -> dict[str, Any]:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    counts: dict[str, int] = {}
    processed = 0
    skipped_existing = 0
    errors: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if record.asr_confidence is not None and record.speech_mix != "untagged":
            skipped_existing += 1
            continue
        if limit is not None and processed >= limit:
            break
        try:
            audio, _sample_rate = load_audio(record.resolved_audio_path(manifest_path))
            segments, _info = model.transcribe(
                audio,
                language="hi" if record.language == "hin" else "en",
                beam_size=3,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            materialized = list(segments)
            text = " ".join(segment.text.strip() for segment in materialized).strip()
            if materialized:
                average_logprob = sum(segment.avg_logprob for segment in materialized) / len(
                    materialized
                )
                confidence = float(max(0.0, min(1.0, math.exp(average_logprob))))
            else:
                confidence = 0.0
            speech_mix = classify_speech_mix(
                text,
                record.language,
                asr_confidence=confidence,
            )
            counts[speech_mix] = counts.get(speech_mix, 0) + 1
            updated = record.model_copy(
                update={
                    "asr_text": text,
                    "asr_confidence": confidence,
                    "speech_mix": speech_mix,
                }
            )
            records[index] = updated.model_copy(
                update={"sampling_weight": category_weight(updated)}
            )
            processed += 1
        except Exception as exc:  # Keep a long tagging job resumable around bad rows.
            errors.append({"id": record.id, "error": repr(exc)})
        if (processed + len(errors)) % checkpoint_every == 0:
            write_manifest(records, output_path)
    write_manifest(records, output_path)
    return {
        "processed": processed,
        "skipped_existing": skipped_existing,
        "remaining_untagged": sum(record.speech_mix == "untagged" for record in records),
        "counts": counts,
        "errors": errors[:1_000],
        "output": str(output_path),
    }


def tag_manifest_with_asr(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    limit: int | None = None,
    checkpoint_every: int = 250,
) -> dict[str, Any]:
    records = _resumable_records(manifest_path, output_path)
    if all(record.speech_mix != "untagged" for record in records):
        return {
            "processed": 0,
            "skipped_existing": len(records),
            "remaining_untagged": 0,
            "counts": {},
            "errors": [],
            "output": str(output_path),
        }
    model = _load_asr_model(model_name, device, compute_type)
    return _tag_with_model(
        model,
        manifest_path,
        output_path,
        records,
        limit=limit,
        checkpoint_every=checkpoint_every,
    )


def tag_prepared_splits_with_asr(
    data_dir: str | Path,
    *,
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    splits: tuple[str, ...] = ("train", "validation", "test"),
    checkpoint_every: int = 250,
) -> dict[str, Any]:
    directory = Path(data_dir)
    jobs: list[tuple[str, Path, Path, list[AudioRecord]]] = []
    for split in splits:
        source = directory / f"{split}.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"Missing prepared manifest: {source}")
        output = directory / f"{split}.tagged.jsonl"
        jobs.append((split, source, output, _resumable_records(source, output)))
    if all(all(record.speech_mix != "untagged" for record in records) for *_, records in jobs):
        return {split: {"processed": 0, "remaining_untagged": 0} for split, *_ in jobs}
    model = _load_asr_model(model_name, device, compute_type)
    return {
        split: _tag_with_model(
            model,
            source,
            output,
            records,
            limit=None,
            checkpoint_every=checkpoint_every,
        )
        for split, source, output, records in jobs
    }


def tag_records_from_transcripts(records: Iterable[AudioRecord]) -> list[AudioRecord]:
    tagged: list[AudioRecord] = []
    for record in records:
        speech_mix = classify_speech_mix(
            record.asr_text or "",
            record.language,
            asr_confidence=record.asr_confidence,
        )
        updated = record.model_copy(update={"speech_mix": speech_mix})
        tagged.append(updated.model_copy(update={"sampling_weight": category_weight(updated)}))
    return tagged
