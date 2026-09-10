# Project AEGIS — Test Suite & Quality Assurance

> **Location:** `tests/`  
> **Framework:** Pytest ($8.0+$) with type checking and coverage profiling  
> **Coverage:** End-to-end testing of DSP preprocessing, network fetchers, SOTA loss math, training loops, and streaming inference

---

## 1. Overview & Test Organization

The `tests/` directory houses an automated testing suite designed to ensure absolute numerical stability, DSP compliance, and real-time streaming latency targets.

```
tests/
├── test_preprocessor.py                  # 10-step ITU-R BS.1770-4 DSP pipeline
├── test_fetchers.py                      # Network reachability across all 10 datasets
├── test_augmentor.py                     # Physics-grounded augmentation invariants
├── test_mixer.py                         # Multi-branch mixture synthesis
├── test_exporter.py                      # WebDataset sharding and DataLoader streaming
├── test_verifier.py                      # Pipeline auditor and markdown compliance report
├── test_multires_loss.py                 # SOTA losses: Multi-Res, Speech-Gated SDR, Distillation, IS³, A-weighted
├── test_training_sync.py                 # Data-Forge <-> Training taxonomy & shard sync
├── test_training_loop.py                 # Complete trainer lifecycle, EMA, worst-class guard
├── test_rev3_ml_inference_upgrades.py   # Lookahead delay buffer, QAT, speech floor, asymmetric hysteresis, lazy unload
├── test_backend_subsystem.py             # Multi-user session manager (<50KB state), batched inference, async transport
├── test_latency_regression.py            # Latency attribution and real-time streaming constraints
└── test_sih_*.py                         # SIH mission-critical acceptance and real data compliance
```

---

## 2. Test Execution Commands

Run tests using pytest:

### 2.1 Run Full Test Suite
```bash
pytest tests/ -v --tb=short
```

### 2.2 Run Specific Subsystems
```bash
# Data Forge: Preprocessor DSP tests
pytest tests/test_preprocessor.py -v

# Data Forge: Live network endpoint reachability checks
pytest tests/test_fetchers.py -v

# Training: SOTA Loss functions & mathematical properties
pytest tests/test_multires_loss.py -v

# Training: Training loop lifecycle, EMA, and checkpoint selection
pytest tests/test_training_loop.py -v

# Inference: Real-time streaming, escalation router, and ONNX Runtime
pytest tests/test_inference_suite.py -v
```

### 2.3 Fast Smoke Test (Excludes Network Probing)
```bash
pytest tests/ -k "not test_fetchers" -v
```

---

## 3. Key Numerical Assertions Verified

- **Sample Rate Uniformity**: Confirms 100% of processed audio is 48,000 Hz.
- **True Peak Limiting**: Asserts all audio is bounded to $\le -1.0$ dBFS without clipping.
- **Physical Invariance**: Confirms `ForbiddenPitchShiftError` triggers if pitch-shifting is attempted on machinery or speech.
- **Loss Monotonicity**: Asserts that `SDRLoss`, `ImpulseWeightedLoss`, and `LocalSnrLoss` decrease monotonically as audio reconstruction quality improves.
- **Zero Lookahead Causality**: Asserts that Model 1 chunk outputs depend strictly on past frames with zero future leakage.
- **Click-Free Handover**: Verifies that router model transitions maintain continuous waveform phase without energy spikes.
