# Deploy the Gradio demo to Hugging Face Spaces

The demo runs on CPU and downloads the dynamic INT8 model when the Space starts. No GPU is needed
for the measured inference path.

## Create the Space

Create a new Gradio Space in the `Mayank022` account. Keep it private while the model repository and
distribution terms remain private. Select Python 3.12 and CPU Basic hardware.

Copy [`SPACE_README.md`](../SPACE_README.md) into the Space as its root `README.md`. The metadata
pins Gradio 6.17.3 and selects `app.py` as the entry point.

Upload these repository paths to the Space:

```text
app.py
requirements.txt
pyproject.toml
LICENSE
src/
```

The Space does not need training configs, prepared audio, checkpoints, W&B files, or evaluation
artifacts.

## Configure the private model

Open the Space's **Settings → Variables and secrets** section.

Add this secret:

```text
HF_TOKEN=<a Hugging Face token with read access to the private model>
```

Add this variable:

```text
HINGLISH_TURN_MODEL=Mayank022/hinglish-turn-detector-whisper-tiny-dual-scale
```

Do not put the token in `app.py`, `requirements.txt`, the README, or a Git commit. Spaces exposes
secrets to the server process as environment variables, and `huggingface_hub` uses `HF_TOKEN` when
the model snapshot is downloaded.

## Run locally first

```bash
uv sync --extra runtime --extra demo
```

```bash
HINGLISH_TURN_MODEL=Mayank022/hinglish-turn-detector-whisper-tiny-dual-scale \
uv run python app.py
```

Open `http://127.0.0.1:7860`, record one complete utterance, and compare it with an unfinished
phrase containing a pause or filler. Stopping a recording or completing an upload runs inference
automatically. The Analyze button remains available for reruns.

The demo also fetches nine deterministic Hindi/English examples from the pinned public Pipecat
training snapshot into the machine's temporary directory. These clips are not committed to the
GitHub or Space repository. If the public dataset viewer is unavailable, the app starts without the
examples and microphone/upload inference continues to work.

## Expected model download

`TurnDetector` requests only:

```text
hinglish-turn.int8.onnx
policy.json
turn_detector_config.json
```

The FP32 ONNX file, PyTorch checkpoint, evaluation reports, and training artifacts are not required
by the Space runtime.

## API

The Analyze button exposes a public Gradio endpoint named `/predict`. A client can inspect the
generated schema from the Space's **Use via API** link. If the Space is private, callers need a
Hugging Face token with access to that Space.

## Troubleshooting

- A `401` or repository-not-found error usually means the `HF_TOKEN` secret is missing or lacks
  access to the private model.
- If the Space says `No ONNX model found`, confirm that the model repository contains
  `hinglish-turn.int8.onnx` on its main revision.
- If dependency installation fails, confirm `requirements.txt`, `pyproject.toml`, and `src/` were
  all uploaded.
- If the example row says the clips are unavailable, confirm that the Space can reach
  `datasets-server.huggingface.co`; the model itself does not depend on that service.
- CPU Basic is sufficient for one request at a time. The demo queue intentionally uses a concurrency
  limit of one so several simultaneous recordings do not contend for the same ONNX session.
