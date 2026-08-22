---
title: Turn Detection · Whisper Tiny Dual-Scale
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.17.3
python_version: 3.12
app_file: app.py
pinned: false
short_description: Audio-only COMPLETE/HOLD detection for Hindi, English, fillers, and pauses.
license: other
---

# Turn Detection Model

Record or upload one user turn. The demo returns a calibrated probability that the speaker has
finished, together with the deployed `COMPLETE` / `HOLD` policy decision.

The model uses the Whisper Tiny audio encoder and a dual-scale attention head. It consumes audio
only at inference time; no transcript is generated.

## Measured E6 result

| Metric | Value |
|---|---:|
| Test examples | 21,995 |
| F1 | 0.7399 |
| AUROC | 0.9361 |
| Overall false-cutoff rate | 4.72% |
| Causal turn-level false-cutoff rate | 4.97% |
| Mean endpoint latency | 512.3 ms |
| CPU end-to-end p95 | 82.36 ms |
| Dynamic INT8 size | 10.16 MiB |

Hindi false cutoffs were 11.44%, compared with 3.62% for English. The source metadata has no
human-verified Hinglish label, so the release does not claim a measured Hinglish-specific score.

## Space configuration

Set the following in the Space settings:

- Secret: `HF_TOKEN` with read access to the private model repository.
- Variable: `HINGLISH_TURN_MODEL=Mayank022/hinglish-turn-detector-whisper-tiny-dual-scale`.

The repository does not contain model weights, source audio, or API keys. The app downloads only
the ONNX model, policy, and model configuration needed for inference. Nine pinned training presets
are fetched from the public dataset viewer into temporary runtime storage; they are not bundled in
this Space repository.
