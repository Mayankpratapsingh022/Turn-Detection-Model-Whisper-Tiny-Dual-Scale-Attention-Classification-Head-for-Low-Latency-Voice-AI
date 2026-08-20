# HinglishTurn-8M

A small, audio-native end-of-turn detector for Hindi, Hinglish, Indian English, filler words, and real pauses. It answers one question: **did the user finish, or are they going to continue?**

The production model is designed to be an approximately 8–10 MB INT8 ONNX file that runs only after a lightweight VAD observes a candidate pause. It does not require a transcript at inference time.

> Status: the repository, data pipeline, model, trainer, export path, and evaluation harness are implemented. No cloud job is launched by this repository. Model weights and measured results must be produced by running the documented training workflow.

The reasoning and promotion rules are written separately in [`reports/METHODOLOGY.md`](reports/METHODOLOGY.md); generated numbers belong in [`reports/REPORT_TEMPLATE.md`](reports/REPORT_TEMPLATE.md).

## Why this is not just another VAD

An energy VAD can tell that the microphone became quiet. It cannot distinguish:

- “mujhe ek cab book karni hai” — probably complete;
- “mujhe ek cab book karni hai, umm…” — probably holding the floor;
- “number is nine eight seven…” — likely an unfinished sequence;
- a genuine ending from a long thinking pause.

HinglishTurn uses Whisper-Tiny's audio representation for linguistic information and a dedicated final-window branch for cadence and fillers. A calibrated policy then trades endpoint latency against false interruptions.

## Scope and data rules

Training audio comes only from the supplied Smart Turn v3.2 dataset family:

- [`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
- [`pipecat-ai/smart-turn-data-v3.2-test`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test)

The configuration rejects every training language except:

- `hin` — Hindi;
- `eng` — English.

No other-language audio can silently enter the pipeline. ASR is used offline only to find high-confidence code-mixed examples; transcripts and language labels never enter production inference.

Before a final run, replace the `null` dataset and base-model revision fields in the chosen YAML with reviewed commit hashes. Until then, audit reports explicitly mark the source revision as `main (unpinned)`.

The upstream dataset card does not currently declare an explicit dataset license. Confirm the terms for derived model weights before publishing them, and do not copy source audio into this repository.

## Architecture

```text
16 kHz mono PCM
       │
       ▼
Silero/energy VAD: candidate pause after 200 ms
       │
       ▼
Last 8 s, left padded, standardized trailing silence
       │
       ▼
80-bin Whisper log-Mel features + valid-frame mask
       │
       ▼
Whisper-Tiny encoder (four layers, 384 hidden size)
       ├── masked global attention pooling
       └── final 1.5 s attention + mean + max pooling
                                │
                                ▼
                   LayerNorm → 256 → 64 → logit
                                │
                                ▼
                         calibrated P(EOT)
```

The encoder is initialized from `openai/whisper-tiny`; existing Smart Turn endpoint weights are not used to initialize the submitted model. The classifier adds a small auxiliary filler head during training, then discards it for deployment.

## Installation

Python 3.11 or 3.12 is required. The project pins Python 3.12 for `uv`.

```bash
uv sync --extra dev
uv run pytest
uv run turn-detector validate-config --config configs/default.yaml
```

Install capabilities as needed:

```bash
# Dataset preparation and offline Hinglish tagging
uv sync --extra data

# GPU/CPU training
uv sync --extra data --extra train --extra tracking --extra eval

# ONNX export
uv sync --extra train --extra export

# Lightweight ONNX runtime (no PyTorch)
uv sync --extra runtime

# Pinned public Smart Turn v3.2 comparison
uv sync --extra baselines

# Gradio app
uv sync --extra train --extra export --extra eval --extra demo

# Hugging Face cache and model publishing commands
uv sync --extra hub
```

## RunPod: setup to packaged model

Use a RunPod image with CUDA 12.x, Python 3.11/3.12, and a persistent volume mounted at
`/workspace`. An A100 40 GB is the reference setup; an A40, L40S, or RTX 4090 also works with a
smaller physical batch if needed. Reserve at least 150 GB, preferably 200 GB, for Hugging Face
snapshots, prepared FLAC, checkpoints, exports, and W&B offline files.

Clone the repository onto the persistent volume and bootstrap it:

```bash
bash scripts/runpod_bootstrap.sh
```

The script creates `.env` from `.env.example` without overwriting an existing file. Fill these
values in `.env`:

```dotenv
HF_TOKEN=hf_...
HF_MODEL_REPO=your-name/hinglish-turn-8m
WANDB_API_KEY=...
WANDB_ENABLED=true
WANDB_ENTITY=your-wandb-user-or-team
WANDB_PROJECT=hinglish-turn-detector
HF_HOME=/workspace/cache/huggingface
WANDB_DIR=/workspace/artifacts/wandb
```

`.env` is git-ignored. Tokens are read from the environment only and are not put into resolved
configs, training reports, cache manifests, release manifests, or W&B run configuration.

Prefetch the two supplied datasets and Whisper Tiny:

```bash
uv run turn-detector cache-assets --config configs/runpod.yaml
```

The command uses the standard Hugging Face cache, resumes partial downloads, and records the
resolved snapshot revisions in `artifacts/cache_manifest.json`. The pipeline immediately derives
pinned RunPod, E5, and E6 YAML files from that manifest, so preparation and training use the exact
cached revisions. The full pipeline is:

```bash
bash scripts/runpod_pipeline.sh
```

It runs preparation, E5 training, hard-negative mining, E6 training, static
INT8 export, calibration, the full evaluation suite, baseline comparison, and release packaging.
It deliberately stops before uploading anything. Restart at a named stage after an interruption:

```bash
PIPELINE_START_AT=train_e5 bash scripts/runpod_pipeline.sh
PIPELINE_START_AT=evaluate PIPELINE_STOP_AFTER=package bash scripts/runpod_pipeline.sh
```

Valid stages are `cache`, `prepare`, `train_e5`, `mine`, `train_e6`, `export`, `calibrate`,
`evaluate`, `baselines`, and `package`. Training itself starts a new optimizer run, so restart a
training stage only when you intend to retrain; dataset downloads reuse the Hugging Face cache.

Every long-running stage emits timestamped `START`, `CHECKPOINT`, `COMPLETE`, or `FAILED` events and
live progress bars that remain visible through tmux and in the `tee` log. Preparation reports rows
seen, accepted parents, derived pause records, language exclusions, rejects, rate, elapsed time, and
ETA. Training reports batches, optimizer steps, loss components,
learning rates, validation progress, early-stopping state, and checkpoint selection. Evaluation,
robustness scoring, baseline scoring, quantization calibration, and pipeline stage durations use the
same progress contract. Set `TURN_DETECTOR_PROGRESS=false` only when machine-readable logs are needed.

W&B logs main/filler/combined losses, both learning rates, causal validation operating points,
selection score, final metrics, runtime configuration, and the best checkpoint as a model artifact.
Set `WANDB_MODE=offline` if the pod temporarily has no network, then run `wandb sync` later.

See [`docs/RUNPOD.md`](docs/RUNPOD.md) for sizing, monitoring, and the exact post-training commands.

## Data preparation

### Local bounded audit

The current development machine cannot cache the complete dataset, so use a streamed audit locally:

```bash
uv run turn-detector data audit \
  --config configs/default.yaml \
  --limit 1000
```

### Full preparation

Run this on a machine with at least 100–150 GB persistent storage:

```bash
uv run turn-detector data prepare --config configs/default.yaml
```

For each Hindi/English clip, preparation performs:

1. decoding and 16 kHz mono resampling;
2. empty-speech, duration, clipping, finite-value, and quality checks;
3. deterministic speech-boundary detection;
4. removal of arbitrary trailing silence;
5. addition of a standardized 200 ms candidate pause;
6. eight-second truncation and left padding;
7. exact waveform hashing and coarse acoustic fingerprinting;
8. duplicate-group-safe train/validation assignment;
9. causal `HOLD` crops at internal pauses with later speech;
10. versioned JSONL manifests and audit reports.

Prepared audio is stored as FLAC. Manifest paths are relative, so moving the complete prepared directory preserves reproducibility.

### Optional offline Hinglish audit

The source already provides the supervised fields used by the model: `endpoint_bool`, `midfiller`,
`endfiller`, `language`, and `synthetic`. Training therefore consumes the prepared manifests
directly and does not require ASR. The source has no separate Hinglish field, so an optional ASR
audit remains available for exploratory code-switch analysis:

```bash
uv sync --extra asr
uv run turn-detector data tag-all \
  --data-dir artifacts/data \
  --model large-v3 \
  --device cuda \
  --compute-type float16
```

This optional command loads faster-whisper once, checkpoints every 250 rows, and safely resumes.
Its pseudo-labels are not ground truth, are not fed to the turn detector, and are not part of the
standard RunPod pipeline. Do not run a full-corpus ASR audit merely to begin training.

Inspect any manifest with:

```bash
uv run turn-detector data summary artifacts/data/train.jsonl
```

## Experiments and training

The repository encodes the ablation sequence rather than relying on undocumented notebook state:

| Run | Config | Change |
|---|---|---|
| E0 | evaluation baseline | fixed 500/800/1200/1600 ms silence |
| E1 | public checkpoint | Smart Turn v3.2 baseline |
| E2 | `e2_global.yaml` | global pooling, uniform data |
| E3 | `e3_focused.yaml` | Hindi/English/filler-focused sampling |
| E4 | `e4_dual.yaml` | dual global/final-window pooling |
| E5 | `e5_causal_filler.yaml` | causal pauses and filler auxiliary loss |
| E6 | `e6_hard_negative.yaml` | mined false-cutoff examples |
| E7 | export | INT8 ONNX deployment model |

Run an experiment:

```bash
uv run turn-detector train \
  --config configs/experiments/e5_causal_filler.yaml
```

Mine incomplete examples that the model incorrectly considers complete:

```bash
uv run turn-detector mine-hard-negatives \
  --model-path artifacts/checkpoints/e5_causal_filler/best \
  --manifest artifacts/data/train.jsonl \
  --output artifacts/data/hard_negatives.jsonl \
  --config configs/default.yaml
```

Then run E6. Checkpoints contain model weights, the exact model/encoder configuration, optimizer and scheduler state, global step, validation metrics, and resolved YAML.

Training defaults:

- BF16 on one A100;
- effective batch 256;
- encoder LR `1e-5`, head LR `1e-4`;
- four epochs maximum with grouped validation;
- encoder frozen for the first 500 optimizer steps;
- `HOLD` errors weighted 2×;
- checkpoint selection by lowest validation endpoint delay at no more than 5% turn-level false cutoffs, with TPR-at-5%-FPR as a fallback when causal rows are unavailable.

W&B is disabled in the default local configuration and enabled by `configs/runpod.yaml` or
`WANDB_ENABLED=true`. A fixed `WANDB_RUN_ID` resumes only the W&B logging stream; it does not resume
optimizer/data-loader state.

## Export and inference

Export, verify, and quantize:

```bash
uv run turn-detector export \
  --checkpoint artifacts/checkpoints/e6_hard_negative/best \
  --output artifacts/export/hinglish-turn.onnx \
  --config configs/default.yaml \
  --quantize
```

By default, export performs static QDQ INT8 quantization using up to 1,024 held-out validation examples; use `--dynamic` only as a fallback. Export fails if PyTorch-to-ONNX FP32 probability error reaches `0.01`. It writes FP32 and INT8 files, the exact model contract, `policy.json`, and a parity/size report.

Fit temperature, threshold, candidate-pause delay, and fallback timeout on validation only:

```bash
uv run turn-detector calibrate \
  --model-path artifacts/export/hinglish-turn.int8.onnx \
  --config configs/default.yaml \
  --target-false-cutoff-rate 0.05
```

This updates only the deployment `policy.json` beside the exported model and writes a separate calibration report. The test split is never used for policy selection.

### Package and publish model weights

After export and evaluation, stage only the deployable weights and evidence (never source audio,
optimizer state, or `.env`):

```bash
uv run turn-detector package-model \
  --checkpoint artifacts/checkpoints/e6_hard_negative/best \
  --export-dir artifacts/export/e6 \
  --evaluation-dir artifacts/evaluation/e6 \
  --output artifacts/release/e6
```

Review `artifacts/release/e6/README.md`, `release_manifest.json`, and all generated metrics. The
upstream dataset card currently lacks an explicit license, so confirm its terms before uploading
derived weights. Publishing is a separate, explicit action and creates a private model repo by
default:

```bash
uv run turn-detector push-model \
  --folder artifacts/release/e6 \
  --repo-id your-name/hinglish-turn-8m \
  --private \
  --acknowledge-source-license-review
```

Use `--public` only after the model card, data terms, and reported measurements have been reviewed.

Score a file:

```bash
uv run turn-detector predict \
  --model-path artifacts/export/hinglish-turn.int8.onnx \
  example.wav
```

Python API:

```python
from turn_detector.inference import TurnDetector

detector = TurnDetector.from_pretrained("artifacts/export/hinglish-turn.int8.onnx")
prediction = detector.score(audio, sample_rate=16_000)

for pcm_chunk in microphone_chunks:
    event = detector.process_chunk(pcm_chunk, sample_rate=16_000)
    if event is not None:
        print(event.as_dict())
```

The ONNX contract is:

- `input_features`: float32 `[batch, 80, 800]`;
- `frame_mask`: int64 `[batch, 800]`;
- `p_complete`: float32 `[batch, 1]`.

## Evaluation

Run the full in-domain suite:

```bash
uv run turn-detector evaluate \
  --model-path artifacts/export/hinglish-turn.int8.onnx \
  --config configs/default.yaml \
  --robustness
```

This produces more than a toy accuracy number:

- classification F1, balanced accuracy, AUROC, average precision;
- false-cutoff and false-hold rates;
- Brier score and expected calibration error;
- Hindi, English, high-confidence Hinglish, filler, source, duration, real/synthetic slices;
- causal internal-pause versus final-endpoint decisions;
- threshold/action-delay/timeout sweep;
- latency at fixed 5% and 10% false-cutoff budgets;
- a complete Pareto frontier;
- grouped bootstrap confidence intervals;
- a turn-group bootstrap interval for the deployed false-cutoff/latency policy;
- clean, telephone, μ-law, noise, reverb, speed, gain, and clipping robustness;
- model-only and end-to-end p50/p95/p99 latency.

The key metric is **endpoint delay at a fixed false-cutoff budget**, not raw accuracy. A fixed timeout can avoid interruptions simply by making every response slow.

Run the paired in-domain baselines after calibration:

```bash
uv run turn-detector compare-baselines \
  --model-path artifacts/export/hinglish-turn.int8.onnx \
  --config configs/default.yaml
```

This compares identical test rows against fixed 500/800/1200/1600 ms timeouts and the official `smart-turn-v3.2-cpu.onnx` pinned to its v3.2 release commit. It reports slice metrics, deployed causal policies, McNemar's paired test, and a paired group-bootstrap F1 delta. It performs no test-set threshold tuning.

### Independent LiveKit EOT Bench

Install [`livekit/eot-bench`](https://github.com/livekit/eot-bench), point the adapter at the model, and evaluate only Hindi and English:

```bash
export HINGLISH_TURN_MODEL="$PWD/artifacts/export/hinglish-turn.int8.onnx"

eot-harness predict \
  --path livekit/eot-bench-data \
  --name all \
  --split validation \
  --adapter turn_detector.evaluation.livekit_adapter:HinglishTurnAudioAdapter \
  --output-dir artifacts/eot-bench
```

The adapter explicitly rejects unsupported language codes. External benchmark audio is evaluation-only and never enters training, ASR tagging, calibration, or threshold selection.

### CPU latency

```bash
uv run turn-detector benchmark \
  --model-path artifacts/export/hinglish-turn.int8.onnx \
  --audio-path example.wav \
  --iterations 1000
```

The acceptance target is an INT8 model no larger than 10 MB and warm p95 model inference below 100 ms on a fixed four-vCPU x86 machine.

## Gradio demo

```bash
uv run turn-detector demo \
  --model-path artifacts/export/hinglish-turn.int8.onnx
```

The demo supports microphone and file input, displays the waveform/candidate pause, and returns probability, decision, recommended wait, and measured model latency.

For a Hugging Face Gradio Space, include the exported ONNX/config/policy files, set `HINGLISH_TURN_MODEL` to the ONNX path, and use [`app.py`](app.py) as the entry point. No Space or model repository is created automatically.

## Cloud execution

[`infra/modal_app.py`](infra/modal_app.py) defines reproducible Modal jobs but creates and runs nothing on import. After explicit cost approval:

```bash
modal volume create hinglish-turn-data
modal run infra/modal_app.py::prepare
modal run infra/modal_app.py::train_model
modal run infra/modal_app.py::evaluate_and_export
```

For the documented ablations, run `train_experiment` for E2 through E5, mine from E5, then run E6. The functions are definitions only and do not launch until you execute these commands:

```bash
modal run infra/modal_app.py::train_experiment --experiment e2_global
modal run infra/modal_app.py::train_experiment --experiment e3_focused
modal run infra/modal_app.py::train_experiment --experiment e4_dual
modal run infra/modal_app.py::train_experiment --experiment e5_causal_filler
modal run infra/modal_app.py::mine_hard_negatives
modal run infra/modal_app.py::train_experiment --experiment e6_hard_negative
modal run infra/modal_app.py::evaluate_and_export --experiment e6_hard_negative
```

Recommended allocation:

- one A100 40 GB or 80 GB;
- 16 CPU cores;
- 48–64 GB RAM;
- 150 GB persistent volume;
- approximately 8–16 A100-hours for experiments, mining, export, and full evaluation.

Expected wall-clock depends mainly on how many Hindi/English rows survive audit and on GPU
throughput. A practical sequence is roughly 1–3 hours for streamed preparation, 45–120 minutes per
Whisper-Tiny experiment, and 1–3 hours for mining/export/full evaluation. These are planning
estimates, not measurements from this repository.

## Reproducibility and tests

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=turn_detector
```

Tests cover configuration constraints, audio boundaries, causal pauses, corruptions, code-mix heuristics, resumable ASR tagging, exact and near-duplicate-safe splits (including cross-repository train/test checks), sampling mass, manifest round trips, classification/calibration metrics, policy sweeps and paired bootstraps, preparation fixtures, and a tiny random Whisper forward pass when ML dependencies are installed.

## Honest limitations

- The source dataset does not identify Hinglish directly; code-mix labels are ASR-derived and must be reported as such.
- Speaker identity is unavailable, so splitting uses duplicate groups and source strata rather than verified speaker-disjoint identities.
- Much of the supplied Hindi data is synthetic; all reports must separate real and synthetic performance.
- The model handles endpointing, not backchannel-versus-barge-in classification or multi-speaker diarization.
- Whisper ignores a feature attention mask internally; the mask is applied to global/tail pooling, while standardized silence controls padding artifacts.
- A model is promoted only when it improves the latency/false-cutoff frontier. If the public Smart Turn baseline wins, the report must say so.
