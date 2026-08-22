from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import turn_detector.demo as demo_module
from turn_detector.config import PolicyConfig
from turn_detector.demo import _format_result, _prepare_audio
from turn_detector.demo_examples import DATASET_REPO, DATASET_REVISION, load_demo_examples
from turn_detector.types import TurnDecision, TurnPrediction


def test_prepare_audio_converts_integer_stereo_to_float_mono() -> None:
    stereo = np.asarray([[32767, -32768], [16384, 16384]], dtype=np.int16)
    sample_rate, waveform = _prepare_audio((16_000, stereo))

    assert sample_rate == 16_000
    assert waveform.dtype == np.float32
    assert waveform.shape == (2,)
    assert np.all(np.isfinite(waveform))
    assert float(np.max(np.abs(waveform))) <= 1.0
    assert waveform[1] == pytest.approx(0.5, abs=1e-4)


def test_prepare_audio_rejects_missing_and_empty_audio() -> None:
    with pytest.raises(ValueError, match="Record or upload"):
        _prepare_audio(None)
    with pytest.raises(ValueError, match="empty"):
        _prepare_audio((16_000, np.zeros(0, dtype=np.float32)))


def test_format_result_renders_complete_and_hold_policy() -> None:
    complete = TurnPrediction(0.82, TurnDecision.COMPLETE, 300, 41.2)
    hold = TurnPrediction(0.21, TurnDecision.HOLD, 1_000, 43.8)

    complete_result = _format_result(
        complete,
        threshold=0.38,
        temperature=2.5522,
        duration_seconds=2.5,
    )
    hold_result = _format_result(
        hold,
        threshold=0.38,
        temperature=2.5522,
        duration_seconds=3.25,
    )

    assert complete_result[0] == "COMPLETE — respond"
    assert complete_result[1] == pytest.approx(82.0)
    assert "300 ms" in complete_result[2]
    assert complete_result[3]["decision"] == "complete"
    assert hold_result[0] == "HOLD — keep listening"
    assert hold_result[1] == pytest.approx(21.0)
    assert "1000 ms" in hold_result[2]


def test_prepare_audio_loads_a_gradio_filepath(tmp_path: Path) -> None:
    path = tmp_path / "recording.wav"
    waveform = np.zeros(8_000, dtype=np.float32)
    sf.write(path, waveform, 16_000)

    sample_rate, loaded = _prepare_audio(str(path))

    assert sample_rate == 16_000
    assert loaded.shape == (8_000,)


def test_build_demo_exposes_working_predict_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("gradio")

    class FakeDetector:
        policy = PolicyConfig(
            threshold=0.38,
            temperature=2.5522,
            min_silence_ms=300,
            timeout_ms=1_000,
        )

        def __init__(self, _model_path: str) -> None:
            pass

        def score(self, _waveform: np.ndarray, _sample_rate: int) -> TurnPrediction:
            return TurnPrediction(0.82, TurnDecision.COMPLETE, 300, 41.2)

    monkeypatch.setattr(demo_module, "TurnDetector", FakeDetector)
    app = demo_module.build_demo("fake-model.onnx", demo_examples=[])
    config = app.get_config_file()

    assert config["title"] == "Turn Detection Model"
    assert any(dependency.get("api_name") == "predict" for dependency in config["dependencies"])
    tab_labels = [
        component["props"]["label"]
        for component in config["components"]
        if component["type"] == "tabitem"
    ]
    assert tab_labels == ["Pipecat presets", "Upload", "Record"]
    progress_holds = [
        dependency
        for dependency in config["dependencies"]
        if str(dependency.get("api_name", "")).startswith("hold_processing_bar")
    ]
    assert len(progress_holds) == 3

    callbacks = {
        function.api_name: function.fn
        for function in app.fns.values()
        if function.api_name in {"predict", "predict_recording", "predict_preset"}
    }
    assert set(callbacks) == {"predict", "predict_recording", "predict_preset"}

    no_audio = callbacks["predict"](None)
    assert no_audio[0] == "NO AUDIO"
    assert no_audio[3]["error"] == "invalid_audio"

    path = tmp_path / "recording.wav"
    sf.write(path, np.zeros(8_000, dtype=np.float32), 16_000)
    for predict in callbacks.values():
        result = predict(str(path))
        assert result[0] == "COMPLETE — respond"
        assert result[1] == pytest.approx(82.0)


def test_demo_examples_download_pinned_hindi_and_english_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = []
    metadata = (
        (33, "hin", False),
        (43, "hin", False),
        (7, "hin", True),
        (79, "hin", False),
        (74, "hin", True),
        (30, "eng", False),
        (76, "eng", True),
        (4, "eng", False),
        (1, "eng", True),
    )
    for row_index, language, endpoint in metadata:
        rows.append(
            {
                "row_idx": row_index,
                "row": {
                    "language": language,
                    "endpoint_bool": endpoint,
                    "audio": [
                        {
                            "src": (
                                "https://datasets-server.huggingface.co/assets/"
                                f"{DATASET_REPO}/--/{DATASET_REVISION}/--/default/train/"
                                f"{row_index}/audio/audio.wav"
                            )
                        }
                    ],
                },
            }
        )
    document = json.dumps({"dataset": DATASET_REPO, "rows": rows}).encode()

    def fake_request(url: str, **_kwargs: object) -> bytes:
        return document if "first-rows" in url else b"RIFF" + (b"\x00" * 64)

    monkeypatch.setattr("turn_detector.demo_examples._request_bytes", fake_request)
    examples = load_demo_examples(tmp_path)

    assert len(examples) == 9
    assert {example.path.name for example in examples} == {
        "row-1-eng.wav",
        "row-4-eng.wav",
        "row-7-hin.wav",
        "row-30-eng.wav",
        "row-33-hin.wav",
        "row-43-hin.wav",
        "row-74-hin.wav",
        "row-76-eng.wav",
        "row-79-hin.wav",
    }
    assert all(example.path.is_file() for example in examples)
    assert any("filler" in example.label for example in examples)
    assert {example.label.split(" · ")[0] for example in examples} == {"Hindi", "English"}
    assert all("HOLD" not in example.label for example in examples)
    assert all("COMPLETE" not in example.label for example in examples)


def test_demo_examples_fail_soft_when_dataset_viewer_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_request(_url: str, **_kwargs: object) -> bytes:
        raise OSError("network unavailable")

    monkeypatch.setattr("turn_detector.demo_examples._request_bytes", fail_request)

    assert load_demo_examples(tmp_path) == []
