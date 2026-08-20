from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from turn_detector.inference import TurnDetector


def build_demo(model_path: str | Path) -> Any:
    try:
        import gradio as gr
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The demo requires `uv sync --extra demo --extra eval`") from exc

    detector = TurnDetector(model_path)

    def predict(audio_value: tuple[int, np.ndarray] | None) -> tuple[dict[str, Any], Any]:
        if audio_value is None:
            raise gr.Error("Record or upload audio first.")
        sample_rate, waveform = audio_value
        waveform = np.asarray(waveform)
        if waveform.size == 0:
            raise gr.Error("The audio is empty.")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if np.issubdtype(waveform.dtype, np.integer):
            waveform = waveform.astype(np.float32) / max(1, np.iinfo(waveform.dtype).max)
        prediction = detector.score(waveform, sample_rate)
        seconds = np.arange(waveform.size) / sample_rate
        figure, axis = plt.subplots(figsize=(10, 2.7))
        axis.plot(seconds, waveform, linewidth=0.6, color="#2357D9")
        axis.axvspan(
            max(0.0, seconds[-1] - detector.policy.min_silence_ms / 1_000),
            seconds[-1],
            color="#F4B942",
            alpha=0.25,
            label="candidate pause",
        )
        axis.set_title(
            f"{prediction.decision.value.upper()} — P(complete)={prediction.probability:.3f}"
        )
        axis.set_xlabel("Time (seconds)")
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.15)
        axis.legend(loc="upper right")
        figure.tight_layout()
        return prediction.as_dict(), figure

    with gr.Blocks(title="HinglishTurn-8M") as demo:
        gr.Markdown(
            "# HinglishTurn-8M\n"
            "Audio-only end-of-turn detection for Hindi, Hinglish, fillers, and pauses. "
            "The model sees no transcript."
        )
        with gr.Row():
            audio = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="User turn",
            )
            output = gr.JSON(label="Decision")
        plot = gr.Plot(label="Audio decision view")
        submit = gr.Button("Detect end of turn", variant="primary")
        submit.click(predict, inputs=audio, outputs=[output, plot])
        gr.Markdown(
            "Try an unfinished phrase such as **`mujhe ek cab book karni hai, umm...`** "
            "and compare it with a complete request."
        )
    return demo
