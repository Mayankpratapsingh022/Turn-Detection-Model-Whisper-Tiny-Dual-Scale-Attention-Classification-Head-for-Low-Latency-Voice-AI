---
language:
- hi
- en
pipeline_tag: audio-classification
tags:
- turn-detection
- semantic-vad
- hinglish
- onnx
license: other
license_name: pending-source-data-review
---

# Whisper Tiny Dual-Scale Turn Detector

Audio-only end-of-turn detection for Hindi/English speech, fillers, and pauses. The intended domain
includes Hinglish conversations, but the source dataset has no human-verified Hinglish label.

## Intended use

Run after a lightweight VAD observes 200 ms of candidate silence. The output is a calibrated probability that the user has completed the current turn.

## Training data

Training contains only `hin` and `eng` examples from Smart Turn v3.2. Disclose exact
real/synthetic, filler, causal-pause, duplicate-group, and split counts from the generated audit.
Do not report a Hinglish-specific count unless it comes from human-verified annotations.

## Metrics

The default `package-model` command generates `README.md` directly from the evaluation,
calibration, export, and baseline JSON artifacts. Publish false cutoffs, endpoint delay at 5%/10%
false-cutoff budgets, slice metrics, stratified robustness, model-only and end-to-end CPU latency,
model size, and confidence intervals. Never fill missing values with estimates.

## Limitations

There is no measured Hinglish-specific score because the source has no verified code-switch field.
Hindi and filler results are useful proxies, not Hinglish ground truth. The detector is audio-only
and does not use conversational text context. It is not a backchannel or barge-in classifier.
