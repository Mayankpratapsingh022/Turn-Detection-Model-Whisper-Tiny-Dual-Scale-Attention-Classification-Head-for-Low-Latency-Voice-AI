# HinglishTurn-8M Technical Report

> The completed E6 report is available in [`FINAL_REPORT.md`](FINAL_REPORT.md). Keep this file as a
> blank structure for future runs rather than overwriting measured release results.

## Claim and operating constraint

State the exact model selected, model size, CPU, p50/p95 latency, and the fixed false-cutoff budget used to rank models. Do not call the model state of the art unless an independent benchmark supports it.

## Data audit

Report Hindi/English counts, real/synthetic composition, filler counts, excluded audio, duplicate
groups, causal pauses, and all split rules. Include the generated audit JSON rather than estimating
these values. State explicitly that the source has no human-verified Hinglish field.

## Experiments

| Run | Architecture/data | Params | INT8 MB | FCR @ 300 ms | FCR @ 600 ms | Latency @ 5% FCR | Hindi F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 | Fixed VAD | – | – | TBD | TBD | TBD | – |
| E1 | Smart Turn v3.2 | TBD | TBD | TBD | TBD | TBD | TBD |
| E2–E7 | See resolved config | TBD | TBD | TBD | TBD | TBD | TBD |

## Robustness and calibration

Report clean, telephone, codec, noise, reverb, clipping, speed, ECE, Brier score, PyTorch/ONNX parity, and FP32/INT8 deltas.

## Failure analysis

Break errors down by incomplete fillers, complete statements containing fillers, source,
synthetic/real status, language, duration, and causal/original example type. Do not infer semantic
categories such as lists or corrections without verified annotations.

## Limitations

Document the absence of Hinglish ground truth, synthetic-data bias, missing speaker identity, the
energy-VAD fallback, the audio-only context limit, and any dataset-license uncertainty.
