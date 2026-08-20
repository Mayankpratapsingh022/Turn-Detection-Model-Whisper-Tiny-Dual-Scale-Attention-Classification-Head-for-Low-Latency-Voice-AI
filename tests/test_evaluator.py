from turn_detector.data.records import AudioRecord
from turn_detector.evaluation.evaluator import (
    _robustness_sampling_summary,
    _stratified_robustness_sample,
)


def _record(
    record_id: str,
    *,
    language: str,
    endpoint: bool,
    kind: str = "original",
    midfiller: bool = False,
    synthetic: bool = False,
) -> AudioRecord:
    return AudioRecord(
        id=record_id,
        parent_id=record_id,
        audio_path=f"{record_id}.flac",
        source_repo="fixture/repo",
        language=language,
        endpoint_bool=endpoint,
        midfiller=midfiller,
        endfiller=False,
        synthetic=synthetic,
        example_kind=kind,
        duration_seconds=1.0,
        valid_samples=16_000,
        speech_seconds=0.8,
        speech_ratio=0.8,
        peak_dbfs=-3.0,
        rms_dbfs=-12.0,
        clipping_ratio=0.0,
        silence_ratio=0.2,
        pause_duration_ms=700 if kind == "causal_internal_pause" else None,
        audio_hash=record_id,
        acoustic_fingerprint=record_id,
        duplicate_group=record_id,
    )


def test_robustness_sample_is_deterministic_and_covers_rare_strata() -> None:
    records = [
        _record(f"eng-{index}", language="eng", endpoint=bool(index % 2)) for index in range(100)
    ]
    records.extend(
        [
            _record("hin-hold", language="hin", endpoint=False),
            _record(
                "hin-causal-filler",
                language="hin",
                endpoint=False,
                kind="causal_internal_pause",
                midfiller=True,
                synthetic=True,
            ),
            _record("hin-complete", language="hin", endpoint=True, synthetic=True),
        ]
    )

    first = _stratified_robustness_sample(records, limit=8, seed=17)
    second = _stratified_robustness_sample(records, limit=8, seed=17)

    assert [row.id for row in first] == [row.id for row in second]
    assert {row.language for row in first} == {"hin", "eng"}
    assert any(row.example_kind == "causal_internal_pause" for row in first)
    assert any(row.midfiller for row in first)
    summary = _robustness_sampling_summary(records, first, requested_limit=8, seed=17)
    assert summary["selected_count"] == 8
    assert summary["selected"]["language"]["hin"] >= 1
    assert summary["composite_strata_selected"] > 1
