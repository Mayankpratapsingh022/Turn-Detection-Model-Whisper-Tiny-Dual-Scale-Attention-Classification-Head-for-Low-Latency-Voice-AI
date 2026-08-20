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

# HinglishTurn-8M

Audio-only end-of-turn detection for Hindi, Hinglish, Indian English, fillers, and pauses.

## Intended use

Run after a lightweight VAD observes 200 ms of candidate silence. The output is a calibrated probability that the user has completed the current turn.

## Training data

Complete this section from `audit.json`. Training must contain only `hin` and `eng` examples from Smart Turn v3.2. Disclose the exact real/synthetic and high-confidence code-mix counts.

## Metrics

Publish false cutoffs at 300/600 ms, endpoint delay at 5%/10% false-cutoff budgets, slice metrics, calibration, robustness, CPU latency, model size, and confidence intervals.

## Limitations

The code-mix tag is derived using ASR and token heuristics because the source dataset has no Hinglish field. The detector is audio-only and does not use conversational text context. It is not a backchannel or barge-in classifier.
