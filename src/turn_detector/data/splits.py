from __future__ import annotations

import hashlib
from collections import defaultdict

from turn_detector.data.records import AudioRecord


def _stratum(record: AudioRecord) -> tuple[object, ...]:
    return (
        record.language,
        record.endpoint_bool,
        bool(record.midfiller),
        bool(record.endfiller),
        record.synthetic,
        record.source_dataset,
    )


def assign_grouped_stratified_splits(
    records: list[AudioRecord],
    *,
    validation_fraction: float = 0.05,
    seed: int = 42,
) -> list[AudioRecord]:
    groups: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        groups[record.duplicate_group].append(record)

    strata: dict[tuple[object, ...], list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        strata[_stratum(members[0])].append(group_id)

    validation_groups: set[str] = set()
    for group_ids in strata.values():
        ordered = sorted(
            group_ids,
            key=lambda group_id: hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest(),
        )
        target = round(len(ordered) * validation_fraction)
        if len(ordered) >= 10:
            target = max(1, target)
        validation_groups.update(ordered[:target])

    return [
        record.model_copy(
            update={
                "split": "validation" if record.duplicate_group in validation_groups else "train"
            }
        )
        for record in records
    ]


def assert_no_group_leakage(records: list[AudioRecord]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        existing = seen.setdefault(record.duplicate_group, record.split)
        if existing != record.split:
            raise ValueError(
                f"Duplicate group {record.duplicate_group} crosses {existing} and {record.split}"
            )
