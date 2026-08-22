from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from turn_detector.audio import ensure_float32_mono, load_audio
from turn_detector.demo_examples import DemoExample, load_demo_examples
from turn_detector.inference import TurnDetector
from turn_detector.types import TurnDecision, TurnPrediction

DEMO_CSS = """
body, .gradio-container {
    background: #050505 !important;
    color: #f5f5f5 !important;
}
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
    border-color: #262626 !important;
    background: #101010 !important;
}
.gradio-container input,
.gradio-container textarea {
    border-color: #303030 !important;
    background: #0a0a0a !important;
    color: #f5f5f5 !important;
}
.gradio-container button.primary {
    border-color: #f5f5f5 !important;
    background: #f5f5f5 !important;
    color: #050505 !important;
}
.gradio-container button.primary:hover {
    border-color: #d4d4d4 !important;
    background: #d4d4d4 !important;
}
.gradio-container button[role="tab"][aria-selected="true"] {
    border-color: #e5e5e5 !important;
    color: #ffffff !important;
}
.td-title {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 4px 0 8px;
}
.td-title h1 {
    margin: 0;
    color: #ffffff;
    font-size: 30px;
}
.td-icon {
    width: 25px;
    height: 25px;
    color: #e5e5e5;
}
.td-option-heading {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 6px;
    color: #f5f5f5;
    font-weight: 700;
}
.td-option-heading .td-icon {
    width: 18px;
    height: 18px;
}
.td-subtitle {
    color: #b8b8b8;
    margin: 0 0 14px;
    padding: 5px 0 3px;
    line-height: 1.5;
}
.td-detect {
    min-height: 46px;
    font-weight: 700;
}
.td-option-note {
    color: #a3a3a3;
    font-size: 13px;
}
.td-preset-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 10px 0 12px;
}
.td-preset-legend span {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 9px;
    border: 1px solid #303030;
    border-radius: 999px;
    color: #c7c7c7;
    font-size: 12px;
}
.td-preset-legend i {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.td-legend-mid { background: #d4ad45; }
.td-legend-end { background: #d77a45; }
.td-legend-speech { background: #64aa7b; }
#td-presets .gallery-item {
    transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
}
#td-presets .gallery-item:hover {
    transform: translateY(-1px);
}
#td-presets .gallery-item:nth-child(-n+3) {
    border-color: #735c1f !important;
    background: #181305 !important;
}
#td-presets .gallery-item:nth-child(n+4):nth-child(-n+5) {
    border-color: #743b20 !important;
    background: #190d08 !important;
}
#td-presets .gallery-item:nth-child(n+6) {
    border-color: #315b40 !important;
    background: #09150e !important;
}
.td-latency-note {
    margin-top: 12px;
    padding: 13px 15px;
    border: 1px solid #292929;
    border-radius: 10px;
    background: #0d0d0d;
    color: #bdbdbd;
    font-size: 13px;
}
.td-progress-wrap {
    padding: 10px 2px 2px;
}
.td-progress-label {
    margin-bottom: 7px;
    color: #a3a3a3;
    font-size: 13px;
    font-weight: 600;
}
.td-progress-track {
    height: 7px;
    overflow: hidden;
    border-radius: 999px;
    background: #262626;
}
.td-progress-fill {
    width: 38%;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #525252, #f5f5f5, #737373);
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.22);
    animation: td-progress 900ms ease-in-out infinite;
}
@keyframes td-progress {
    from { transform: translateX(-110%); }
    to { transform: translateX(290%); }
}
"""

TITLE_HTML = """
<div class="td-title">
  <svg class="td-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <rect x="9" y="2" width="6" height="12" rx="3"></rect>
    <path d="M5 10a7 7 0 0 0 14 0M12 17v5M8 22h8"></path>
  </svg>
  <h1>Turn Detection Model</h1>
</div>
"""

RECORD_ICON_HTML = """
<div class="td-option-heading">
  <svg class="td-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <rect x="9" y="2" width="6" height="12" rx="3"></rect>
    <path d="M5 10a7 7 0 0 0 14 0M12 17v5"></path>
  </svg>
  Record from microphone
</div>
"""

UPLOAD_ICON_HTML = """
<div class="td-option-heading">
  <svg class="td-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M12 16V4M7 9l5-5 5 5M4 20h16"></path>
  </svg>
  Upload an audio file
</div>
"""

PRESET_ICON_HTML = """
<div class="td-option-heading">
  <svg class="td-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M5 3l14 9-14 9V3z"></path>
  </svg>
  Use a Pipecat preset
</div>
"""

READY_DETAILS = """
Record with the microphone or upload an audio file, then click **Detect turn**.

- **COMPLETE** means the assistant may respond.
- **HOLD** means the speaker may continue.
"""

PROCESSING_BAR_HTML = """
<div class="td-progress-wrap" role="status" aria-live="polite">
  <div class="td-progress-label">Processing audio and calculating probability…</div>
  <div class="td-progress-track"><div class="td-progress-fill"></div></div>
</div>
"""

MODEL_DETAILS = """
### Model information

- Whisper Tiny audio encoder with a dual-scale classification head
- Audio-only inference; no transcription is generated
- Uses the last 8 seconds of 16 kHz mono audio
- Dynamic INT8 ONNX runtime
- Test F1: **0.7399** · AUROC: **0.9361** · false-cutoff rate: **4.72%**
"""


def _prepare_audio(audio_value: Any) -> tuple[int, np.ndarray]:
    """Normalize either a Gradio filepath or the legacy NumPy audio tuple."""

    if audio_value is None:
        raise ValueError("Record or upload audio first.")

    if isinstance(audio_value, tuple) and len(audio_value) == 2:
        sample_rate, raw_waveform = audio_value
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("The audio sample rate is invalid.")
        raw = np.asarray(raw_waveform)
        if np.issubdtype(raw.dtype, np.integer):
            limits = np.iinfo(raw.dtype)
            if np.issubdtype(raw.dtype, np.unsignedinteger):
                midpoint = (float(limits.max) + 1.0) / 2.0
                raw = (raw.astype(np.float32) - midpoint) / midpoint
            else:
                scale = max(abs(limits.min), limits.max)
                raw = raw.astype(np.float32) / float(scale)
        waveform = ensure_float32_mono(raw)
    else:
        waveform, sample_rate = load_audio(audio_value)

    if waveform.size == 0:
        raise ValueError("The audio is empty.")
    return int(sample_rate), waveform


def _format_result(
    prediction: TurnPrediction,
    *,
    threshold: float,
    temperature: float,
    duration_seconds: float,
) -> tuple[str, float, str, dict[str, Any]]:
    probability_percent = float(prediction.probability * 100.0)
    if prediction.decision is TurnDecision.COMPLETE:
        decision = "COMPLETE — respond"
        explanation = "The speaker sounds finished, so the assistant may take the turn."
    else:
        decision = "HOLD — keep listening"
        explanation = "The speaker may continue after the pause, so keep the turn open."

    details = (
        f"**{explanation}**\n\n"
        f"Input: `{duration_seconds:.2f} s` · Inference: `{prediction.inference_ms:.1f} ms` · "
        f"Threshold: `{threshold:.2f}` · Recommended wait: "
        f"`{prediction.recommended_wait_ms} ms`"
    )
    raw_result = prediction.as_dict()
    raw_result.update(
        {
            "threshold": threshold,
            "temperature": temperature,
            "input_duration_seconds": round(duration_seconds, 4),
            "model": "Whisper Tiny dual-scale dynamic INT8",
        }
    )
    return decision, probability_percent, details, raw_result


def demo_theme() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The demo requires `uv sync --extra demo --extra runtime`") from exc
    return gr.themes.Base(
        primary_hue="gray",
        secondary_hue="gray",
        neutral_hue="gray",
        radius_size="md",
    ).set(
        body_background_fill="#050505",
        body_background_fill_dark="#050505",
        block_background_fill="#101010",
        block_background_fill_dark="#101010",
        body_text_color="#f5f5f5",
        body_text_color_dark="#f5f5f5",
        border_color_primary="#292929",
        border_color_primary_dark="#292929",
        button_primary_background_fill="#f5f5f5",
        button_primary_background_fill_dark="#f5f5f5",
        button_primary_text_color="#050505",
        button_primary_text_color_dark="#050505",
    )


def demo_launch_kwargs() -> dict[str, Any]:
    return {
        "theme": demo_theme(),
        "css": DEMO_CSS,
        "show_error": True,
        "footer_links": ["api"],
        "ssr_mode": False,
    }


def build_demo(
    model_path: str | Path,
    *,
    demo_examples: Sequence[DemoExample] | None = None,
) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The demo requires `uv sync --extra demo --extra runtime`") from exc

    detector = TurnDetector(model_path)
    resolved_examples = list(demo_examples) if demo_examples is not None else load_demo_examples()
    mid_filler_examples = [
        example for example in resolved_examples if "mid-filler" in example.label
    ]
    end_filler_examples = [
        example for example in resolved_examples if "end-filler" in example.label
    ]
    speech_examples = [
        example
        for example in resolved_examples
        if example not in mid_filler_examples and example not in end_filler_examples
    ]
    ordered_examples = mid_filler_examples + end_filler_examples + speech_examples

    def predict(audio_value: Any) -> tuple[str, float, str, dict[str, Any]]:
        try:
            sample_rate, waveform = _prepare_audio(audio_value)
        except (OSError, ValueError, TypeError) as exc:
            return (
                "NO AUDIO",
                0.0,
                f"Error: {exc}",
                {"error": "invalid_audio", "message": str(exc)},
            )

        try:
            prediction = detector.score(waveform, sample_rate)
        except (OSError, ValueError, RuntimeError) as exc:
            return (
                "ERROR",
                0.0,
                f"Error: could not process this audio: {exc}",
                {"error": "inference_failed", "message": str(exc)},
            )

        return _format_result(
            prediction,
            threshold=detector.policy.threshold,
            temperature=detector.policy.temperature,
            duration_seconds=waveform.size / sample_rate,
        )

    # ZeroGPU requires the real Gradio inference callback to declare its GPU lease. The deployed
    # dynamic INT8 graph still uses ONNX Runtime's CPU provider; wrapping the actual callback keeps
    # local CPU behavior unchanged while satisfying the hosting contract on ZeroGPU.
    try:
        import spaces
    except ImportError:  # pragma: no cover - provided only by Hugging Face ZeroGPU
        pass
    else:
        predict = spaces.GPU(duration=10)(predict)

    def show_processing() -> str:
        return PROCESSING_BAR_HTML

    def hide_processing() -> str:
        return ""

    def hold_processing_bar() -> None:
        # Give the browser enough time to paint the custom progress bar before very fast inference
        # replaces it with the final result. This does not alter the audio or model prediction.
        time.sleep(0.45)

    def clear() -> tuple[None, None, None, str, str, float, str, dict[str, Any]]:
        return None, None, None, "", "Waiting for audio", 0.0, READY_DETAILS, {}

    with gr.Blocks(title="Turn Detection Model", analytics_enabled=False) as demo:
        gr.HTML(TITLE_HTML)
        gr.Markdown(
            "Whisper Tiny + dual-scale attention for Hindi/English turns, fillers, and pauses.",
            elem_classes="td-subtitle",
        )

        gr.Markdown("### 1. Choose an audio source")
        with gr.Tabs():
            with gr.Tab("Pipecat presets"):
                gr.HTML(PRESET_ICON_HTML)
                if resolved_examples:
                    gr.Markdown(
                        "Choose a pinned training example, listen to it, then detect the turn.",
                        elem_classes="td-option-note",
                    )
                    preset_audio = gr.Audio(
                        type="filepath",
                        label="Selected preset",
                        interactive=False,
                    )
                    gr.HTML(
                        '<div class="td-preset-legend">'
                        '<span><i class="td-legend-mid"></i>Mid-filler</span>'
                        '<span><i class="td-legend-end"></i>End-filler</span>'
                        '<span><i class="td-legend-speech"></i>Speech and pauses</span>'
                        "</div>"
                    )
                    gr.Examples(
                        examples=[[str(example.path)] for example in ordered_examples],
                        example_labels=[example.label for example in ordered_examples],
                        inputs=preset_audio,
                        cache_examples=False,
                        label="Pipecat training examples",
                        elem_id="td-presets",
                        api_visibility="private",
                    )
                    detect_preset = gr.Button(
                        "Detect preset turn",
                        variant="primary",
                        elem_classes="td-detect",
                    )
                    gr.Markdown(
                        "Preset audio is fetched from the pinned public Pipecat training snapshot "
                        "into temporary runtime storage. Preset metadata is used only to choose "
                        "diverse clips and color categories; the colors do not represent the "
                        "expected decision. Every displayed probability and decision is computed "
                        "live by the ONNX model from the selected audio."
                    )
                else:
                    preset_audio = gr.Audio(
                        type="filepath",
                        label="Selected preset",
                        interactive=False,
                        visible=False,
                    )
                    gr.Markdown(
                        "Preset clips are temporarily unavailable. Recording and upload still work."
                    )
                    detect_preset = gr.Button(
                        "Detect preset turn",
                        interactive=False,
                        elem_classes="td-detect",
                    )

            with gr.Tab("Upload"):
                gr.HTML(UPLOAD_ICON_HTML)
                gr.Markdown(
                    "Drag and drop a WAV or MP3 file below, or click the box to browse.",
                    elem_classes="td-option-note",
                )
                uploaded_audio = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    format="wav",
                    label="Drop audio here",
                )
                detect_upload = gr.Button(
                    "Detect uploaded turn",
                    variant="primary",
                    elem_classes="td-detect",
                )

            with gr.Tab("Record"):
                gr.HTML(RECORD_ICON_HTML)
                gr.Markdown(
                    "Record a short user turn, stop the recording, then detect it.",
                    elem_classes="td-option-note",
                )
                recorded_audio = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    format="wav",
                    label="Microphone recording",
                )
                detect_recording = gr.Button(
                    "Detect recorded turn",
                    variant="primary",
                    elem_classes="td-detect",
                )

        processing_bar = gr.HTML(value="", show_label=False)
        gr.Markdown(
            "**Why recording can take longer:** a new microphone recording must be finalized by "
            "the browser, encoded, and uploaded before inference starts. Presets are already "
            "downloaded on the server, so they feel nearly instant. Both paths run the same "
            "Whisper encoder and ONNX turn-detection model.",
            elem_classes="td-latency-note",
        )

        gr.Markdown("### 2. Result")
        with gr.Row():
            decision = gr.Textbox(
                value="Waiting for audio",
                label="Decision",
                interactive=False,
            )
            probability = gr.Number(
                value=0.0,
                label="Completion probability (%)",
                interactive=False,
                precision=2,
            )
        details = gr.Markdown(READY_DETAILS)

        reset = gr.Button("Reset all")

        with gr.Accordion("Raw model output", open=False):
            raw_output = gr.JSON(value={}, label=None)

        with gr.Accordion("Model details", open=False):
            gr.Markdown(MODEL_DETAILS)

        def connect_detection(button: Any, audio_input: Any, api_name: str) -> None:
            processing_event = button.click(
                show_processing,
                inputs=None,
                outputs=processing_bar,
                queue=False,
            )
            paint_event = processing_event.then(
                hold_processing_bar,
                inputs=None,
                outputs=None,
                queue=False,
            )
            prediction_event = paint_event.then(
                predict,
                inputs=audio_input,
                outputs=[decision, probability, details, raw_output],
                api_name=api_name,
                api_description="Predict COMPLETE or HOLD for one audio file.",
                show_progress="full",
                show_progress_on=[decision, probability, details],
                queue=True,
            )
            prediction_event.then(
                hide_processing,
                inputs=None,
                outputs=processing_bar,
                queue=False,
            )

        connect_detection(detect_recording, recorded_audio, "predict_recording")
        connect_detection(detect_upload, uploaded_audio, "predict")
        connect_detection(detect_preset, preset_audio, "predict_preset")
        reset.click(
            clear,
            inputs=None,
            outputs=[
                recorded_audio,
                uploaded_audio,
                preset_audio,
                processing_bar,
                decision,
                probability,
                details,
                raw_output,
            ],
            queue=False,
        )

    return demo
