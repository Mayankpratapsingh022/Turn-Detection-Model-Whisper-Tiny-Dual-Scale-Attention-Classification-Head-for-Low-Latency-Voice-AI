# Methodology and experiment contract

## What the model decides

The detector is called only after the VAD sees a candidate pause. Its output is the probability that the current user turn is complete. Silence detection itself is not the learning problem: a fixed VAD can detect quiet but cannot tell a sentence ending from `haan, ek second...`.

The production policy has four separate quantities:

1. candidate-pause duration before inference;
2. calibrated completion probability;
3. completion threshold;
4. maximum fallback timeout.

Keeping these separate matters. A system that never interrupts but waits 2.4 seconds is safe and unusable; a useful comparison must show latency and interruptions together.

## Data contract

Only the Hindi (`hin`) and English (`eng`) rows from the two Smart Turn v3.2 repositories may enter training, validation, calibration, or the in-domain test. `DataConfig` rejects additional languages.

The pipeline decodes to mono 16 kHz, audits bad audio, removes arbitrary terminal silence, adds exactly 200 ms of candidate silence, keeps the last eight seconds, and left-pads shorter turns. It retains exact hashes and coarse acoustic fingerprints. All crops from one utterance share a duplicate group. Test parents overlapping either training split are removed before evaluation.

The upstream rows do not identify Hinglish. The standard pipeline therefore makes no
Hinglish-specific measurement claim. Large-v3 ASR remains an optional exploratory indexing tool,
not a model input or ground-truth label source. Hindi and filler slices are reported directly from
source metadata; they are relevant to the intended domain but are not substitutes for a verified
code-switch benchmark.

Internal pauses are causal hard negatives only when the energy segmentation finds later speech of at least 500 ms. A crop ends 200 ms into that pause and is labeled `HOLD`; it never includes the future audio used to prove that the speaker continued.

## Model choice

The backbone is the encoder half of Whisper Tiny. It is small, pretrained on multilingual acoustics, and already has useful prosodic and linguistic structure. The head uses two views:

- masked attention over the complete valid turn;
- attention, mean, and max pooling over the final 1.5 seconds.

The complete-turn branch carries grammar and long-range context. The final branch carries cadence, hesitation, list continuation, and end fillers. During training, an auxiliary head predicts whether a filler occurs using both global and final embeddings. It is omitted from the ONNX output.

The model is deliberately audio-only at runtime. Training uses source endpoint, filler, language,
synthetic, and source-dataset metadata; optional ASR pseudo-labels are not part of the standard
training path or inference features.

## Experiment sequence

| Run | Question |
|---|---|
| E0 | How much latency is required by fixed 500/800/1200/1600 ms VAD timeouts? |
| E1 | Does the candidate beat the pinned public Smart Turn v3.2 CPU model on identical rows? |
| E2 | What does a plain global Whisper encoder learn with uniform sampling? |
| E3 | Does balanced Hindi/English and filler-focused sampling help the intended slices? |
| E4 | Does the final-window branch improve the latency/interruption frontier? |
| E5 | Do causal pauses and filler supervision reduce false cutoffs? |
| E6 | Does one round of mined high-scoring `HOLD` examples improve the remaining failure set? |
| E7 | What is lost after held-out static INT8 quantization? |

The best checkpoint is selected on validation endpoint delay subject to at most 5% turn-level false cutoffs. Temperature and policy values are fitted on validation after export. The test set is evaluated once with those frozen values.

## Required evaluation

The primary result is mean and p95 endpoint delay at fixed 5% and 10% false-cutoff budgets. The report must also include:

- F1, balanced accuracy, AUROC, average precision, false-hold rate, Brier score, and ECE;
- Hindi, English, filler, source, real/synthetic, duration, and causal-pause slices;
- turn-group bootstrap intervals for deployed false-cutoff rate and endpoint delay;
- paired group-bootstrap delta and McNemar test against Smart Turn v3.2;
- telephone, μ-law, 5/10/20 dB noise, reverb, speed, gain, and clipping stress tests;
- FP32/INT8 parity, model size, and fixed-machine model-only plus preprocessing-and-model
  p50/p95/p99 CPU latency.

An independent Hindi/Hinglish benchmark would strengthen external validity, but it is not reported
unless human-verified labels and a leakage-free dataset are actually available.

No model is called “best” because it has the highest accuracy. It is promoted only if its Pareto
frontier improves and the measured Hindi and filler slices do not regress materially. If the public
baseline wins, that is the result.

## Compute plan

A single A100 40 GB, 16 CPUs, 64 GB RAM, and a 150 GB persistent volume are sufficient.
Preparation should run on CPU; training and hard-negative mining use the GPU; ONNX export,
quantization, and the final CPU benchmark do not need an A100. Full-corpus ASR is not part of the
training path because the source already provides endpoint and filler labels.

## Known limits

The source has Hindi but no human-annotated Hinglish/code-switch field. Speaker identities are
unavailable. Much of the Hindi subset is synthetic. The causal pause generator depends on an energy
segmentation heuristic. Audio-only endpointing cannot resolve every semantic ambiguity and does not
replace diarization, backchannel detection, or barge-in policy.
