# HinglishTurn-8M Technical Report

## Claim and operating constraint

State the exact model selected, model size, CPU, p50/p95 latency, and the fixed false-cutoff budget used to rank models. Do not call the model state of the art unless an independent benchmark supports it.

## Data audit

Report Hindi/English counts, real/synthetic composition, filler counts, excluded audio, duplicate groups, causal pauses, high-confidence Hinglish count, and all split rules. Include the generated `audit.json` rather than estimating these values.

## Experiments

| Run | Architecture/data | Params | INT8 MB | FCR @ 300 ms | FCR @ 600 ms | Latency @ 5% FCR | Hinglish F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 | Fixed VAD | – | – | TBD | TBD | TBD | – |
| E1 | Smart Turn v3.2 | TBD | TBD | TBD | TBD | TBD | TBD |
| E2–E7 | See resolved config | TBD | TBD | TBD | TBD | TBD | TBD |

## Robustness and calibration

Report clean, telephone, codec, noise, reverb, clipping, speed, ECE, Brier score, PyTorch/ONNX parity, and FP32/INT8 deltas.

## Failure analysis

Break errors down by incomplete fillers, complete statements containing fillers, lists, numbers, corrections, questions, long hesitations, source, synthetic/real, and ASR-derived speech-mix label.

## Limitations

Document pseudo-Hinglish label noise, synthetic-data bias, missing speaker identity, the energy-VAD fallback, the audio-only context limit, and any dataset-license uncertainty.

