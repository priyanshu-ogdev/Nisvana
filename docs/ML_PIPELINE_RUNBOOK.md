# AEGIS ML Pipeline Runbook

This document is the implementation-facing description of the complete AEGIS
machine-learning path. It is intentionally precise about what is implemented,
what is selected at runtime, and what still requires measurement on the target
hardware. The system is designed around the SIH/DRDO requirements:

- generalization to changing defence and industrial noise;
- SNR improvement and intelligibility rather than noise suppression alone;
- PESQ > 2.5 and STOI > 0.85 on held-out data;
- real-time streaming with an RTF below 1.0;
- deployability on GPU systems and constrained edge CPUs.

## 1. Pipeline at a glance

```text
real recordings
    -> data_forge fetch / preprocess / split / mix / export / verify
    -> data/shards (WebDataset train, val, test_generalization)
    -> training.scripts.train_pipeline
       -> CleanUMamba cross-check teacher
       -> DeepFilterNet primary student
       -> DeepFilterNet escalation student
       -> acoustic classifier gate
       -> optional/reused AEC model
    -> checkpoint selection and real-data evaluation
    -> stateful ONNX export (explicit recurrent state I/O)
    -> ONNX Runtime CPU/CUDA/TensorRT adapter
    -> streaming router + hybrid ANC + transmission preparation
    -> SIH scorecard and target-hardware latency measurement
```

The data pipeline is run once to produce verified shards. Stage 7 training
consumes those shards directly; it does not call `data_forge`, regenerate
mixtures, or create a second Torch dataset on disk.

## 2. Data contract and generalization

`data_forge/config.py` is the source of truth for the data contract:

| Property | Contract |
|---|---|
| Sample rate | 48,000 Hz |
| Target loudness | -23 LUFS |
| Mixture duration | 4 seconds |
| SE SNR envelope | -5 to +20 dB |
| Split names | `train`, `val`, `test_generalization` |
| Runtime chunk | 480 samples / 10 ms |
| Classifier context | 9,600 samples / 200 ms |

Every speech-enhancement sidecar retains source dataset, unified class, native
sample rate, sync tier, target/measured SNR, and split. Training uses class and
sync-tier weights only for the training split. Validation and
`test_generalization` are never oversampled.

Shard discovery is exact and fails closed. Existing archives named `gentest`
or `test` are accepted as explicit aliases for `test_generalization`; unrelated
archives are never used as a fallback. This prevents train/test leakage and
avoids reporting a generalization result on the wrong data.

The generalization strategy is complementary:

1. preserve real acoustic variation and provenance;
2. balance thin operational classes with bounded sample weights;
3. expose the model to the full measured SNR envelope;
4. use multi-resolution, impulse-aware, speech-presence-aware losses;
5. select checkpoints with worst-class guards, not aggregate quality alone;
6. verify on a held-out noise/environment split.

Metrics from the DeepFilterNet paper or bibliography are not treated as AEGIS
results. AEGIS claims are valid only after evaluation on its own held-out
shards.

## 3. Training design

The unified launcher is `scripts/07_train.sh`, which calls
`training.scripts.train_pipeline`. In `--model all` mode the dependency order
is deliberate:

1. **CleanUMamba cross-check/teacher** is trained first.
2. **Primary enhancement** receives the teacher checkpoint for frozen
   spectrogram distillation.
3. **Escalation enhancement** is trained for difficult/impulsive conditions.
4. **Classifier gate** learns the 200 ms routing categories.
5. **AEC** is reused by default because `train_by_default=False`.

Each model has an isolated checkpoint and log directory. This prevents one
stage from overwriting another stage's `best_checkpoint.pt`.

### Objective and optimization

The enhancement objective combines multi-resolution spectral reconstruction,
waveform/SI-SNR terms, speech-presence-gated SDR, impulse/onset weighting,
perceptual frequency weighting, mixture consistency, and optional teacher
distillation. The teacher is frozen during student training and contributes no
inference cost.

The GB10 starting configuration is:

```bash
bash scripts/07_train.sh \
  --model all \
  --device cuda \
  --precision bf16 \
  --qat \
  --distillation \
  --batch-size 32 \
  --grad-accum 2 \
  --num-workers 16
```

BF16 is used for convolutional/linear compute. Fallback GRUs execute in FP32
outside AMP because the server's cuDNN rejects this recurrent shape; the
native PyTorch kernel is selected once and cached. This preserves recurrent
stability and avoids repeatedly handling a cuDNN exception.

QAT observers are attached to supported convolutional and linear layers.
Recurrent modules remain unquantized during QAT because their tuple/state
semantics are not safely handled by the generic eager QAT graph.

## 4. Runtime inference design

The runtime has one streaming contract and two execution backends.

### Signal path

```text
mic array / reference / throat mic
    -> MultichannelHardwareFrontend
    -> 480-sample contiguous hop
    -> classifier rolling context (200 ms)
    -> primary or escalation model
    -> intelligibility floor and crossfade
    -> optional NLMS residual cancellation
    -> limiter/transmission preparation
    -> headset/radio output
```

`StatefulHopProcessor` is the correct processor for the implemented causal
models. It forwards non-overlapping 480-sample hops and lets the model carry
its own convolution context and recurrent state. The OLA processor remains
available only for genuinely stateless full-context functions; wrapping the
stateful models in OLA would duplicate samples and corrupt recurrent history.

### Router policy

- **Primary:** default low-latency enhancement path.
- **Escalation:** immediate transition for impulsive or low-SNR conditions.
- **Bypass:** only after 15 consecutive clean classifier decisions.
- **Pipelined route:** optional; removes classifier compute from the current
  chunk's critical path but adds one chunk of decision lag.
- **Crossfade:** transitions use the previous real output, never a second
  probe inference.
- **State safety:** the activated model is reset before the first chunk after a
  bypass interval.
- **Memory:** escalation can be lazy-loaded and unloaded after idle timeout.

Model 3 is a training cross-check/teacher and fallback deployment option; it is
not silently run as a third simultaneous route by the current router. This
keeps the hot path bounded and prevents unnecessary duplicate inference.

### ONNX and edge integration

`inference/engines/onnx_engine.py` exports enhancement models with explicit
state inputs and outputs. `OnnxRuntimeSession` threads those arrays between
calls and selects TensorRT, CUDA, or CPU providers. `OnnxModelAdapter` exposes
the same callable/reset interface expected by the router, so routing logic is
not duplicated for ONNX deployment.

Production entry points now support both backends:

```bash
python -m inference.scripts.enhance_audio \
  --input noisy.wav --output enhanced.wav \
  --model router --backend onnx \
  --onnx-dir data/onnx_models \
  --onnx-provider CUDAExecutionProvider \
  --onnx-provider CPUExecutionProvider

python -m inference.scripts.live_mic_anc \
  --backend onnx \
  --onnx-dir data/onnx_models \
  --onnx-provider CUDAExecutionProvider \
  --onnx-provider CPUExecutionProvider
```

The provider list is ordered by preference. Missing exports, invalid backend
options, and incompatible checkpoint/backend combinations fail explicitly.

## 5. Verification workflow

### Before training

```bash
bash scripts/07_train.sh --model all --device cuda \
  --precision bf16 --qat --distillation \
  --batch-size 32 --grad-accum 2 --num-workers 16 --dry-run
```

Confirm the SE shard directory contains `train`, `val`, and
`test_generalization` or its exact `gentest` alias. Do not rerun data-forge if
the verified shards already exist.

### After training

```bash
bash scripts/13_evaluate_models.sh \
  --model aegis-se-primary --split val \
  --checkpoint training/checkpoints/aegis-se-primary/best_checkpoint.pt

bash scripts/13_evaluate_models.sh \
  --model aegis-se-primary --split test_generalization \
  --checkpoint training/checkpoints/aegis-se-primary/best_checkpoint.pt
```

Evaluation must report real data only, with class-balanced results and
per-SNR/per-class breakdowns where available.

### Export and runtime checks

```bash
bash scripts/15_export_edge_onnx.sh \
  --model se_primary \
  --checkpoint training/checkpoints/aegis-se-primary/best_checkpoint.pt \
  --output data/onnx_models/aegis-se-primary.onnx

bash scripts/18_verify_sih_compliance.sh
```

The SIH scorecard fails closed when latency is not measured. It reports
absolute SNR and delta-SNR, STOI, PESQ, DNSMOS proxy, compute latency,
algorithmic delay, total latency, and RTF. The final pass requires:

```text
SNR > 15 dB or delta-SNR > 15 dB
STOI > 0.85
PESQ > 2.5
RTF < 1.0
```

Dry runs, proxy DNSMOS, bibliography scores, and model initialization do not
establish SIH compliance. Latency must be measured with the exported artifact
on the target CPU/GPU, and quality must be measured on held-out AEGIS data.

## 6. What the upgrades solve

The original runtime risk was not merely hardware speed. A missing model
artifact could silently select a generic fallback, state could be discarded
between chunks, and the documented ONNX path was not reachable from the
application entry points. The current design closes those integration gaps:

- data provenance and split isolation reach the trainer;
- teacher distillation reaches the primary student;
- checkpoints and logs are isolated per model;
- recurrent state is preserved in PyTorch and ONNX;
- classifier, router, enhancement, and hybrid ANC produce one coherent signal;
- edge backends are selectable from the real CLIs;
- SIH reporting distinguishes verified results from estimates.

Still open by design are empirical questions: final quality after full
training, target-device RTF, and the quality/latency trade-off of INT8 versus
FP16. Those are deployment measurements, not values that should be invented
in documentation.

## 7. Frontend/backend synchronization

The dashboard is an observer and control surface, not a source of simulated
health claims. The backend protocol remains authoritative and its Pydantic
schemas are exported into `frontend/src/ws/schemas/`. The frontend must be
able to show, per node:

- the active model, inference backend, and execution provider;
- measured inference latency, algorithmic delay, and real-time factor;
- thermal tier and any explicit degradation reason;
- AEC mode (`disabled`, `placeholder`, or `deepvqe`);
- link state, queue depth, dropped frames, and measured RTT.

When no live node is connected, simulation values are labeled as simulation
and SIH metrics are left unavailable. The UI must never convert a missing
measurement into a passing score or display a placeholder AEC/model as
production-active.

The synchronization path is:

```text
aegis-backend/src/ws/protocol.py
    -> python -m src.ws.protocol --export-schemas
    -> frontend/src/ws/schemas/
    -> useConnectionStore.js
    -> per-node PersonCard + global telemetry bar
```

Future runtime work should preserve this ownership boundary: inference
measures model/backend state, the node publishes it, and the frontend renders
it without recomputing or estimating SIH compliance.
