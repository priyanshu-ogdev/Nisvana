# AEGIS Machine-Learning Design

This document is the detailed design reference for the AEGIS training and
inference layer. It explains the algorithms, model boundaries, data sources,
hyperparameters, synchronization rules, and SIH acceptance gates. It separates
implemented behavior from target behavior so that a checkpoint, export, or
dashboard can never be mistaken for a measured deployment result.

The shorter operational guide is
[`ML_PIPELINE_RUNBOOK.md`](ML_PIPELINE_RUNBOOK.md). The product requirements
remain in [`Nisvana_PRD.md`](Nisvana_PRD.md).

## 1. Design objectives

AEGIS is optimized for speech intelligibility under rapidly changing defence
and industrial noise, not for maximum noise attenuation in isolation. The
design therefore has five objectives:

1. Preserve speech formants, consonants, and transient speech energy.
2. Generalize across stationary machinery, rotor noise, sirens, wind, gunfire,
   explosions, reverberation, and changing SNR.
3. Process a 48 kHz stream in 480-sample, 10 ms hops with bounded state.
4. Escalate only when the acoustic condition justifies the added cost or
   delay.
5. Fail visibly when a trained/exported artifact or target measurement is
   missing.

The SIH/PRD gates are SNR or delta-SNR above 15 dB, STOI above 0.85, PESQ
above 2.5, and real-time factor below 1.0. These are acceptance criteria, not
guaranteed scores from papers or dry runs.

## 2. End-to-end ML path

```text
real recordings
  -> fetch and provenance
  -> 48 kHz / -23 LUFS / mono normalization
  -> deduplication and strict split assignment
  -> grounded mixing and metadata sidecars
  -> WebDataset tar shards
  -> model-specific training
  -> validation and worst-class checkpoint selection
  -> stateful ONNX export
  -> ONNX Runtime CPU/CUDA/TensorRT adapter
  -> 200 ms classifier gate
  -> stateful primary/escalation enhancement
  -> limiter, optional AEC, and playback
```

Stage 7 reads existing shards. It does not regenerate data, create a second
dataset, or silently substitute arbitrary shards when a required split is
absent.

## 3. Data contract

| Property | Implemented contract |
|---|---|
| Internal sample rate | 48,000 Hz |
| Loudness target | -23 LUFS, ITU-R BS.1770 |
| Training mixture length | 4.0 seconds / 192,000 samples |
| Enhancement SNR | -5 to +20 dB global; gunfire extension to -15 dB |
| Inference hop | 480 samples / 10 ms |
| Gate context | 9,600 samples / 200 ms |
| Required splits | train, val, test_generalization |
| Shard size | 2,048 samples per tar by default |
| Output peak target | -1 dBFS true-peak policy |

The data-forge mixer preserves source dataset, native sample rate, sync tier,
unified class, measured SNR, and provenance. Tier-3 16 kHz material is
upsampled for the 48 kHz contract and receives a bounded 0.25 sampling weight;
it is never allowed to dominate full-band native recordings.

## 4. Datasets and corpus sizes

Dataset profiles are defined in `data_forge/config.py`. The following
classification is the source-to-model mapping. Where a corpus size is known
from the repository's verified bibliography or completed pipeline report it
is stated; otherwise the audit manifest is authoritative and no invented hour
count is used.

| Dataset | Acoustic content | Native rate/tier | Known size | Used by |
|---|---|---:|---:|---|
| NOISEX-92 | tracked vehicles, artillery, jet, naval, gunfire, general noise | mixed / tier 3 | corpus size is manifest-dependent | Models 1-3, Model 4 |
| SHAReD | high-explosive blast waveforms | 48 kHz / tier 1 | 326 recordings | Models 1-3, Model 4 |
| Gunshot Triangulation / Dryad | firearm and field gunshot recordings | 44.1 kHz / tier 2 | source count is manifest-dependent | Models 1-3, Model 4 |
| DroneAudioSet | UAV and rotor noise | 48 kHz / tier 1 | approximately 23.5 hours in the design audit | Models 1-3, Model 4 |
| Military Audio Dataset (MAD) | military vehicles, gunfire, explosions | 48 kHz / tier 1 | 8,075 clips, approximately 12 hours | Models 1-3, Model 4 |
| VoiceBank-DEMAND / CSTR VCTK | clean speech and environmental noise | 48 kHz / tier 1 | manifest-dependent | Models 1-3, Model 4 |
| DNS Challenge | broad clean speech and general noise | 48 kHz / tier 1 | manifest-dependent | Models 1-3, Model 4 |
| AEC Challenge | near-end, far-end, echo and room conditions | 48 kHz / tier 1 | manifest-dependent | Model 5 |
| UrbanSound8K / ESC-50 subset | sirens and wind/environmental material | 44.1 kHz / tier 2 | selected subset; manifest-dependent | Models 1-3, Model 4 |
| OpenSLR-28 RIRs | room impulse responses | 16 kHz / tier 3 | manifest-dependent | Models 1-3 |
| MUSAN | held-out generalization noise | 16 kHz / tier 3 | generalization-only; manifest-dependent | validation/generalization |

The completed pipeline report describes approximately 20,694 raw files,
200,000 speech-enhancement triplets, and 60,000 classifier samples. These
are pipeline-run inventory values, not hard-coded assumptions. Before a
training run, use `python -m data_forge verify` and record the generated
manifest and dataset card. The AEC count must likewise be read from that
audit; the current AEC trainer is disabled by default and uses a verified
checkpoint instead.

### Why the dataset mix generalizes

Generalization comes from independent source provenance and held-out
environments, not from treating every file as interchangeable. Thin classes
are bounded-oversampled (up to 6x), low-SNR cases are reweighted, transient
classes receive impulse-aware loss weighting, and validation is class-balanced
without train-style oversampling. Pitch shifting is forbidden for speech,
machinery, weapons, and impulse-response classes because it corrupts formants,
RPMs, and blast physics.

## 5. Model roster

### 5.1 Model 1: `aegis-se-primary`

**Algorithm:** DeepFilterNet3-style causal spectral enhancement. The model
estimates ERB-band gains and deep-filter coefficients from streaming
time-frequency features and reconstructs speech with the causal analysis /
synthesis path.

**Role:** Always-on default enhancement path.

**Design:**

- pretrained DeepFilterNet3 initialization;
- zero configured lookahead (`df_lookahead=0`,
  `conv_lookahead=0`) for the lowest latency route;
- 32 ERB bands (`nb_erb=32`);
- 96 deep-filter features (`nb_df=96`);
- deep-filter order 5 (`df_order=5`);
- convolution width 64 (`conv_ch=64`);
- explicit recurrent and convolutional context between 480-sample hops;
- QAT for supported convolutional/linear layers, with recurrent state left
  outside generic eager QAT.

**Optimization:**

- AdamW;
- learning rate `5e-4`, minimum `1e-6`;
- two warmup epochs;
- weight decay ramps from `1e-12` to `0.01`;
- batch schedule 32 -> 64 -> 128 at epochs 0, 3, and 10;
- maximum 50 epochs, early stopping patience 12;
- BF16 for convolutional/linear work on GB10;
- GRU execution forced to FP32 when cuDNN rejects the deployed shape.

**Loss:** multi-resolution spectral loss with FFT sizes 256, 512, 1024,
and 2048; complex and magnitude terms at factor 500; gamma 0.3; local-SNR
term `1e-3`; SDR factor 0.5; impulse factor 0.3; onset boost 3.0;
perceptual-frequency factor 0.2; speech-presence SDR boost 2.5 in the
300-4000 Hz speech band. A frozen CleanUMamba teacher contributes a
distillation factor of 0.3.

### 5.2 Model 2: `aegis-se-escalation`

**Algorithm:** the same DeepFilterNet3 family and feature representation as
Model 1, but retains the pretrained stock lookahead configuration
(`df_lookahead=2`, `conv_lookahead=2`).

**Role:** hard-condition route for low SNR, impulsive, or rapidly changing
noise. It trades an explicit one-chunk output delay in the implemented
fallback wrapper for additional context. The vendored DeepFilterNet runtime
must report its own measured filterbank delay; the router never assumes that
hop size equals total algorithmic delay.

**Optimization:** AdamW, learning rate `2e-4`, minimum `1e-6`, one warmup
epoch, batch schedule 64 -> 128 at epochs 0 and 5, maximum 25 epochs,
patience 8. It uses the same enhancement loss stack as Model 1 but samples
low-SNR examples more heavily:
`[-15,-10,-5,0,5,10,15,20]` with weights
`[.12,.14,.18,.18,.14,.10,.08,.06]`.

This is a distribution adaptation, not an architecture rewrite, so the lower
learning rate reduces catastrophic drift from the pretrained model.

### 5.3 Model 3: `aegis-se-crosscheck`

**Algorithm:** causal time-domain encoder -> recurrent bottleneck -> decoder,
with the bottleneck corresponding to the intended pruned CleanUMamba
state-space design. The repository fallback uses a unidirectional GRU-based
causal implementation with a 15-sample encoder kernel, 32 channels, and a
32-unit recurrent bottleneck.

**Role:** teacher for primary distillation, independent quality cross-check,
and thermal fallback when the full DeepFilterNet route is unavailable. It is
not silently run in parallel with the primary route.

**Parameters:** target 1M-pruned class, Adam, learning rate `2e-4`,
betas `(0.9, 0.999)`, 5% linear warmup, cosine decay, batch 16, 150,000
fine-tuning steps, 4-second clips, full STFT loss. The teacher is frozen
when used for student distillation.

The paper's DNS results are not AEGIS results; this model must be evaluated
on the AEGIS 48 kHz held-out shards.

### 5.4 Model 4: `aegis-clf-gate`

**Algorithm:** lightweight two-layer GRU classifier over 64-band log-mel
features. It is intentionally not a speech-enhancement model. A compact
feature classifier is cheaper and more stable for routing than asking the
enhancer to infer its own operating state.

**Role:** 200 ms acoustic gate for three classes:

- `harmonic`: machinery, drones, rotor, siren, wind, and continuous noise;
- `impulsive`: gunshots, explosions, and impacts;
- `speech_dominant`: clean or sufficiently high-SNR speech.

**Parameters:** hidden size 128, two recurrent layers, AdamW, learning rate
`1e-3`, cosine schedule, weight decay `1e-4`, batch 192, maximum 25 epochs,
patience 6, weighted cross entropy with class weights harmonic 1.0,
impulsive 3.0, speech-dominant 1.2. Labels are generated from the same
mixing metadata as the enhancement branch, so the gate and enhancer do not
train on contradictory examples.

### 5.5 Model 5: `aegis-aec-gate`

**Algorithm:** gated DeepVQE-style echo cancellation path. It should consume
the microphone signal and a time-aligned far-end/playback reference, but the
current repository keeps the model disabled by default.

**Role:** suppress far-end echo in non-speech windows without damaging
near-end speech. It is gated by VAD and crossfaded on speech onset/offset.

**Current status:** `train_by_default=False`; the configured
`deepvqe-ggml-unofficial` checkpoint is reused as-is when available. The
backend reports `aec_mode=placeholder` when only the placeholder path is
active, and `deepvqe` only when a real model is loaded. AEC Challenge data
is the intended training source, but no AEC quality claim is valid until
the artifact and playback-reference path are measured.

## 6. Synchronization between models

The models share one data and runtime contract but do not share hidden state.
This prevents a classifier reset, thermal swap, or route transition from
corrupting another model's temporal context.

```text
same 48 kHz / 4 s training contract
        |
        +--> Model 3 teacher -> frozen distillation features -> Model 1
        |
        +--> Model 1 primary ----+
        |                        +--> router -> output
        +--> Model 2 escalation -+
        |
        +--> Model 4 gate -------> route decision only
        |
        +--> Model 5 AEC <------- playback/far-end reference
```

At runtime:

1. The classifier accumulates 200 ms while the audio path advances in 10 ms
   hops.
2. Primary is the default route.
3. Impulsive or low-SNR decisions enter escalation immediately.
4. Bypass requires 15 consecutive clean decisions.
5. Model state is reset before the first chunk after bypass or model swap.
6. Crossfade uses already-produced output and never performs a hidden probe
   inference.
7. The limiter is the final safety boundary.

## 7. Inference implementation

The production edge interface is `OnnxModelAdapter` backed by
`OnnxRuntimeSession`. Exports expose explicit state tensors where required;
the session threads state input/output arrays across calls. Providers are
selected in preference order:

| Platform | Preferred provider | Fallback |
|---|---|---|
| Raspberry Pi | CPUExecutionProvider | explicit heuristic emergency mode |
| Jetson | TensorRTExecutionProvider | CUDAExecutionProvider, then CPU |
| DGX/GB10 | CUDA/TensorRT as available | CPU for verification |

PyTorch is the training and development backend. It remains available in the
CLI, but Pi production should not require Torch. Missing ONNX artifacts must
be visible as degraded state; they must not be renamed RNNoise or presented
as DeepFilterNet output.

The final audio sequence is:

```text
capture -> channel frontend -> reference NLMS
        -> optional harmonic preprocessing
        -> classifier/router
        -> primary/escalation/CleanUMamba
        -> intelligibility floor and crossfade
        -> optional gated AEC
        -> -1 dBFS limiter
        -> transmission preparation / DAC
```

The frontend dashboard receives model, provider, thermal, AEC, latency, RTF,
queue, and degradation fields from the backend protocol. It renders those
values and never computes an SIH pass from placeholders.

## 8. SIH evaluation and release gates

Every trained model is evaluated on both `val` and
`test_generalization`, with per-class and per-SNR results. The release report
must include noisy input SNR, enhanced output SNR, delta-SNR, STOI, PESQ,
segmental SNR, transient recoverability, compute latency, algorithmic delay,
total latency, and RTF.

The release gate is:

```text
output SNR > 15 dB OR delta-SNR > 15 dB
STOI > 0.85
PESQ > 2.5
RTF < 1.0
```

Latency is measured on the actual exported artifact and target hardware.
Bibliography scores, DNSMOS proxies, model initialization time, dry runs,
and synthetic dashboard values cannot pass the gate.

## 9. Known limitations and honest status

- The repository contains a working stateful PyTorch/ONNX inference design,
  but target Pi measurements remain deployment work.
- The current backend fallback is a heuristic/noise-reduction path when
  production artifacts are absent; it is not real RNNoise.
- CleanUMamba's repository fallback is GRU-based and is not a proof of a
  released Mamba SSM checkpoint.
- DeepVQE AEC is not production-active without a real loaded artifact and
  aligned playback reference.
- Raw corpus and AEC sizes must be taken from the generated audit manifest,
  not inferred from configuration names.
- Final SIH compliance is intentionally unclaimed until the completed model
  outputs and target-hardware measurements are attached to a verification
  report.

