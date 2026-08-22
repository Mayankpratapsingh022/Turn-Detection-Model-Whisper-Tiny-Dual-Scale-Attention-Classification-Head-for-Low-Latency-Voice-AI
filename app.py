from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings(
    "ignore",
    message="'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated.*",
)

from turn_detector.demo import build_demo, demo_launch_kwargs  # noqa: E402
from turn_detector.environment import load_project_env  # noqa: E402

load_project_env()

MODEL_PATH = os.environ.get(
    "HINGLISH_TURN_MODEL",
    "Mayank022/hinglish-turn-detector-whisper-tiny-dual-scale",
)

demo = build_demo(MODEL_PATH)

if __name__ == "__main__":
    demo.launch(**demo_launch_kwargs())
