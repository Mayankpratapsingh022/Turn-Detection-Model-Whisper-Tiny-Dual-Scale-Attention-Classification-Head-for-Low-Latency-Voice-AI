from __future__ import annotations

import os

from turn_detector.demo import build_demo

MODEL_PATH = os.environ.get(
    "HINGLISH_TURN_MODEL",
    "artifacts/export/hinglish-turn.int8.onnx",
)

demo = build_demo(MODEL_PATH)

if __name__ == "__main__":
    demo.launch()
