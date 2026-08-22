# Final technical report

## Whisper Tiny dual-scale turn detector, E6 dynamic INT8

This report records the model trained and evaluated on 21 August 2026. It is based on the packaged E6 artifacts, not estimates copied into a template. The corresponding model repository is [Mayank022/hinglish-turn-detector-whisper-tiny-dual-scale](https://huggingface.co/Mayank022/hinglish-turn-detector-whisper-tiny-dual-scale).

## Summary

The system predicts whether a user has completed a speaking turn or is holding the floor. It runs after a VAD finds a candidate pause and uses only audio at inference time. The selected model combines the Whisper Tiny encoder with separate global-context and final-window pooling branches.

The dynamic INT8 release reached 0.7399 F1 and 0.9361 AUROC on 21,995 held-out examples. Its overall false-cutoff rate was 4.72%. The causal deployment policy also stayed within its 5% interruption budget, with a 4.97% turn-level false-cutoff rate and 512.3 ms mean endpoint latency.

The paired public baseline comparison is the strongest result in the run. Under the same validation-selected false-cutoff constraint, the candidate scored 0.7399 F1 and Smart Turn v3.2 scored 0.4364. The +0.3035 difference had a 95% parent-turn bootstrap interval of [+0.2884, +0.3182].

The weakest result is Hindi interruption behavior. Hindi F1 was 0.7656, but the Hindi false-cutoff rate was 11.44%. English false cutoffs were 3.62%. Heavy noise, reverb, and filler-bearing pauses also caused clear regressions. This model is a solid challenge submission and useful prototype; it still needs targeted work before production use in a Hindi-first assistant.

## Task and operating contract

The model is not a VAD. The VAD answers "is the signal quiet?" The turn detector answers "does this pause mean the user is finished?"

Each inference receives up to eight seconds of recent audio ending in a standardized candidate pause. The model returns a probability of `COMPLETE`. The application then applies four validation-selected policy values:

1. the minimum amount of silence before scoring;
2. a temperature for probability calibration;
3. the completion threshold;
4. a fallback timeout when the model predicts `HOLD`.

Keeping model score and timing policy separate prevents a misleading evaluation. A fixed timeout can avoid interruptions by making every response slow. This report measures interruption and latency together.

## Source and split construction

Training and in-domain evaluation used the Hindi (`hin`) and English (`eng`) rows from the provided Smart Turn v3.2 train/test family. The following revisions were resolved and pinned before the full run:

| Asset | Pinned revision |
|---|---|
| Training source | `e564e2ac567f774d1880aa1db6ce97afb8c519b7` |
| Test source | `0500378e8ed6d38e37b016e24d261e8e6c6a6859` |
| `openai/whisper-tiny` | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` |

No language other than Hindi or English entered the standard preparation path. The source fields used for supervision were:

| Field | Use |
|---|---|
| `endpoint_bool` | main `COMPLETE` / `HOLD` target |
| `midfiller` | auxiliary mid-turn filler target |
| `endfiller` | auxiliary end-turn filler target |
| `language` | filtering, sampling, and evaluation slices |
| `synthetic` | audit and evaluation slices |

The source has no human-verified Hinglish or code-switch field. A row marked Hindi may contain some English, but that cannot be treated as measured Hinglish ground truth. No ASR-derived pseudo-label was used as a training target or inference feature.

### Prepared counts

| Split or example type | Count |
|---|---:|
| Parent training utterances | 73,679 |
| Original training examples | 73,679 |
| Causal internal-pause examples | 106,910 |
| E5 training manifest | 180,589 |
| Mined hard negatives added for E6 | 10,933 |
| E6 training examples | 191,522 |
| Validation examples | 9,493 |
| Held-out test examples | 21,995 |
| Hindi test examples | 3,124 |
| English test examples | 18,871 |

### Audio preparation

The preparation pipeline:

- decodes and resamples to mono 16 kHz;
- rejects empty, non-finite, invalid-duration, and heavily clipped audio;
- detects the speech boundary, removes arbitrary terminal silence, and appends a fixed candidate pause;
- keeps the last eight seconds and left-pads short examples;
- records waveform hashes and coarse acoustic fingerprints;
- groups all crops from the same parent before splitting;
- removes train/test parent overlaps;
- writes relocatable JSONL manifests and FLAC audio.

Causal examples are created only when an internal pause has at least 500 ms of later speech. The crop stops inside the pause and excludes the future audio used to prove continuation. This avoids leaking the answer into the model input.

## Model

### Encoder

The backbone is the audio encoder from `openai/whisper-tiny`: four Transformer layers with a hidden size of 384. Input audio becomes 80-bin log-Mel features with 800 time frames. The Whisper convolutional front end reduces this to 400 encoder positions.

### Dual-scale pooling

The head uses two temporal views:

- masked attention pooling over the full valid turn;
- attention, mean, and max pooling over the final 1.5 seconds.

The four 384-dimensional vectors are concatenated into a 1,536-dimensional representation. The main classifier is:

```text
LayerNorm(1536)
  → Linear(1536, 256)
  → GELU
  → Dropout(0.1)
  → Linear(256, 64)
  → GELU
  → Linear(64, 1)
```

A separate filler head consumes the global and tail-attention vectors and predicts two logits: mid-filler and end-filler. This head supplies auxiliary supervision during training and is not required by the exported runtime graph.

| Model property | Value |
|---|---:|
| Total parameters | 8,299,397 |
| Non-encoder parameters | 513,413 |
| Maximum input | 8 s at 16 kHz |
| Tail window | 1.5 s |
| FP32 ONNX size | 33,267,901 bytes |
| Selected INT8 ONNX size | 10,651,569 bytes |

## Training

The main loss is weighted binary cross entropy. A false `COMPLETE` on a `HOLD` example receives 2x weight. The two filler logits use binary cross entropy where filler labels are available, multiplied by 0.15 before addition to the main loss.

The sampler works from the prepared manifest rather than assuming the source ratio. E6 used the following expected sampling mass:

| Sampling dimension | E6 mass |
|---|---:|
| Hindi | 50.0% |
| English | 50.0% |
| COMPLETE | 35.0% |
| HOLD | 65.0% |
| Any filler | 53.4% |
| Causal pause | 44.9% |
| Mined hard negative | 30.0% |

| Optimizer setting | Value |
|---|---:|
| Physical batch | 32 |
| Effective batch | 256 |
| Gradient accumulation | 8 |
| Maximum epochs | 4 |
| Encoder learning rate | 1e-5 |
| Head learning rate | 1e-4 |
| Warmup | 5% |
| Weight decay | 0.01 |
| Precision | BF16 |
| Encoder freeze | first 500 optimizer steps |
| Evaluation and checkpoint interval | 500 optimizer steps |

E5 learned from the original and causal examples. Its best checkpoint then scored the incomplete training set, and the highest-scoring false completions were mined as hard negatives. E6 retrained with those examples contributing 30% of sampling mass.

### Observed compute

The first E5 run silently fell back to CPU because PyTorch had been resolved for CUDA 13.0 while the RunPod driver supported CUDA 12.4. It completed, but took 45,410 seconds. The environment was corrected to PyTorch 2.6.0 + CUDA 12.4, after which the L40S was used normally.

| Stage | Device | Wall time |
|---|---|---:|
| Data preparation | CPU | 5,786 s (1 h 36 m) |
| E5 training | CPU fallback | 45,410 s (12 h 37 m) |
| Hard-negative mining | NVIDIA L40S | 1,198 s (20 m) |
| E6 training | NVIDIA L40S | 1,048 s (17 m 28 s) |
| Dynamic calibration through packaging | CPU | 5,901 s (1 h 38 m) |

The bootstrap script now checks `torch.cuda.is_available()` and reports the installed Torch/CUDA/GPU combination before a training run. A CUDA-capable pod will fail early instead of quietly spending hours on CPU.

## Evaluation design

The test set was not used to select temperature, threshold, silence delay, timeout, or baseline policy. Selection happened on validation under a target 5% false-cutoff budget. The chosen policy was frozen before test scoring.

The evaluation suite includes:

- F1, balanced accuracy, AUROC, average precision, false-cutoff rate, and false-hold rate;
- expected calibration error and Brier score in the machine-readable report;
- language, filler, original/causal, synthetic/real, duration, and source slices;
- a threshold, action-delay, and timeout policy sweep;
- parent-turn grouped bootstrap confidence intervals;
- matched paired comparison against Smart Turn v3.2;
- fixed-timeout baselines;
- telephone, codec, noise, reverb, speed, gain, and clipping stress tests;
- model-only and preprocessing-plus-model CPU latency.

## Held-out results

| Metric | Test value |
|---|---:|
| Examples | 21,995 |
| F1 | 0.7399 |
| Balanced accuracy | 0.8248 |
| AUROC | 0.9361 |
| Average precision | 0.7983 |
| False-cutoff rate | 0.0472 |
| False-hold rate | 0.3033 |
| Expected calibration error | 0.1043 |

The error tradeoff is sensible for a low-interruption policy: 4.72% of `HOLD` examples are cut off, while 30.33% of true completions are initially held. The fallback timeout prevents an indefinite wait.

### Deployed causal policy

| Policy value | Selected value |
|---|---:|
| Threshold | 0.38 |
| Temperature | 2.5522 |
| Minimum silence | 300 ms |
| Fallback timeout | 1,000 ms |
| Turn-level false-cutoff rate | 0.0497 |
| Mean endpoint latency | 512.3 ms |
| Median endpoint latency | 300 ms |
| P95 endpoint latency | 1,000 ms |

The p95 equals the fallback timeout, which means a meaningful tail of completed turns was not accepted by the model and waited for the safety timeout.

### Language and filler slices

| Slice | Count | F1 | False-cutoff rate |
|---|---:|---:|---:|
| Hindi | 3,124 | 0.7656 | 0.1144 |
| English | 18,871 | 0.7340 | 0.0362 |
| Mid-filler | 5,099 | 0.7239 | 0.0840 |
| Any filler | 6,235 | 0.7034 | 0.0773 |
| Original examples | 8,955 | 0.7905 | 0.0648 |

The Hindi aggregate F1 looks good in isolation, but its 11.44% false-cutoff rate is the more relevant endpointing result. It is roughly three times the English rate. A single global threshold meets the aggregate budget partly because English accounts for most test examples.

The end-filler, hard incomplete-filler, and causal internal-pause slices contain only one class. Ordinary binary F1 is undefined there, so the generated model card reports `Not available`. Future reporting should expose class-appropriate rates for these slices: false holds for complete-only slices and false cutoffs for hold-only slices.

## Baseline comparison

Both policies were selected on validation under the same 5% false-cutoff budget and then frozen. Scoring used identical test rows.

| Model | Test F1 |
|---|---:|
| E6 dynamic INT8 candidate | 0.7399 |
| Smart Turn v3.2 | 0.4364 |

| Paired statistic | Value |
|---|---:|
| Candidate minus baseline | +0.3035 F1 |
| 95% group-bootstrap interval | [+0.2884, +0.3182] |
| Probability candidate is better | 1.0 |
| Matched validation false-cutoff budget | 0.05 |

The interval does not cross zero and is narrow enough to rule out a marginal win on this in-domain test. It does not establish superiority on unrelated datasets or live traffic.

## Robustness

The robustness subset was deterministically stratified by language, label, original/causal status, filler type, and synthetic status. Each corruption was applied to the same selected records. The clean-subset score differs from the full-test score because its class and slice mixture is deliberately different.

| Condition | F1 | False-cutoff rate |
|---|---:|---:|
| Clean | 0.7044 | 0.0799 |
| Clipping | 0.6835 | 0.0839 |
| Low gain | 0.6904 | 0.0852 |
| μ-law | 0.7042 | 0.0826 |
| Noise, 20 dB | 0.6905 | 0.0826 |
| Noise, 10 dB | 0.5283 | 0.1358 |
| Noise, 5 dB | 0.3679 | 0.1292 |
| Reverb | 0.5861 | 0.1438 |
| Speed 0.9x | 0.6653 | 0.1012 |
| Speed 1.1x | 0.6608 | 0.0759 |
| Telephone | 0.6477 | 0.1105 |

The model is relatively stable under μ-law, moderate noise, clipping, and low gain. At 10 dB and 5 dB SNR, F1 drops sharply. Reverb produces the highest false-cutoff rate at 14.38%. Noise/reverb augmentation and more real Hindi conversational audio are the clearest next data improvements.

## Export decision

The PyTorch-to-FP32 ONNX conversion was effectively exact. Static QDQ activation quantization met the original size goal but failed probability parity badly. The deployment release therefore uses dynamic weight-only INT8.

| Export | Bytes | MiB | Max probability difference | Mean difference | Parity |
|---|---:|---:|---:|---:|---|
| FP32 ONNX | 33,267,901 | 31.73 | 0.00000095 vs PyTorch | — | Pass |
| Static INT8 QDQ | 9,104,324 | 8.68 | 0.574263 | — | Fail |
| Dynamic INT8 | 10,651,569 | 10.16 | 0.017759 | 0.009105 | Pass |

The dynamic file misses the 10 MiB target by 165,809 bytes, about 1.6%. That is a better trade than shipping an 8.68 MiB model whose probability can move by 0.57 after quantization.

## CPU latency

The runtime used ONNX Runtime with one intra-op thread. The host exposed 128 logical CPUs, but the model session itself was deliberately single-threaded. Disk I/O and audio decoding were excluded.

| Measurement | p50 | p95 | p99 |
|---|---:|---:|---:|
| Model only | 39.15 ms | 75.89 ms | 79.14 ms |
| End to end | 47.36 ms | 82.36 ms | 87.90 ms |

End-to-end timing includes waveform standardization, log-Mel extraction, temperature calibration, and model inference. It does not include the 300 ms candidate silence or the possible 1,000 ms fallback wait.

## What worked

- Causal crops converted ordinary endpoint labels into examples of real continuation after a pause.
- Balanced Hindi/English sampling prevented the larger English pool from dominating every batch.
- The final-window branch gave the classifier a direct view of cadence and hesitation.
- Separate mid-filler and end-filler targets preserved distinctions that an "any filler" label would lose.
- One hard-negative round focused E6 on the exact `HOLD` examples E5 wanted to interrupt.
- Validation-only calibration kept policy tuning out of the test set.
- Dynamic INT8 preserved probability behavior while keeping CPU latency below 100 ms at p99 on the measured host.

## What did not work well

- Static activation quantization was unusable despite meeting the size target.
- A global threshold hid a large Hindi/English false-cutoff gap.
- The source metadata was not enough to support a verified Hinglish-specific score.
- Heavy noise and reverb exposed weak acoustic robustness.
- Complete-only and hold-only slice summaries need class-appropriate metrics instead of `Not available` for every field.
- The original environment allowed an expensive silent CPU fallback; the bootstrap now blocks that failure mode.

## Limits and release position

- This is an in-domain result. No independent human-labeled Hinglish benchmark was available.
- English examples are not guaranteed to be exclusively Indian English.
- Some Hindi audio is synthetic, and speaker identities are unavailable.
- Duplicate protection cannot replace a verified speaker-disjoint split.
- Audio-only classification cannot use transcript meaning, dialog history, gaze, or speaker state.
- The model does not perform diarization, backchannel detection, barge-in arbitration, or safety classification.
- The complete source collection does not state one explicit license. The uploaded model remains private with a pending-license-review notice.

The right release claim is narrow: this is a small audio-only Hindi/English turn detector that beats the selected public baseline on the held-out in-domain test while meeting an aggregate 5% false-cutoff budget. It is not yet a production-certified Hinglish endpointing model.

## Reproduction and evidence

The full pipeline is:

```bash
bash scripts/runpod_bootstrap.sh
bash scripts/runpod_pipeline.sh
```

The release package contains:

- `model.safetensors`;
- `hinglish-turn.onnx`;
- `hinglish-turn.int8.onnx`;
- `turn_detector_config.json`;
- `policy.json` and `calibration_report.json`;
- `training_report.json` and resolved configuration;
- full evaluation and paired baseline reports;
- generated model card and SHA-256 release manifest.

Source audio, `.env`, API keys, and optimizer state are deliberately excluded. Training logs are stored in W&B, while the packaged JSON reports contain the metrics needed to audit the release.
