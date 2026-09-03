# Project AEGIS — training/ Architecture: Complete Design, Tuned & Finalized

This is the design document behind the `training/` package delivered alongside this file. Every hyperparameter in the code is either traced to a primary source (DeepFilterNet3's official `config.ini`, CleanUMamba's paper) or explicitly marked as a reasoned default — never presented with false authority. The code and this document are meant to be read together; this explains the *why*, the code has the *values*.

---

## 1. Naming Conventions (training/ and inference/)

Established once, applied consistently:
- **Model registry keys:** `aegis-<branch>-<role>` — `aegis-se-primary`, `aegis-se-escalation`, `aegis-se-crosscheck`, `aegis-clf-gate`, `aegis-aec-gate`.
- **Checkpoints:** `<model_key>-v<config_version>-step<N>.pt` — the filename alone identifies which config produced it, verified working in the smoke test (`aegis-se-primary-v1-step00012000.pt`).
- **Config classes:** `<ModelKey>Config` in PascalCase, one file per model in `training/configs/`.
- **`inference/` mirrors this exactly** (`inference/engines/se_primary_engine.py` loading `aegis-se-primary-*.pt`), so a checkpoint produced by training has an unambiguous consumer in inference without a lookup table — this was a deliberate design choice, not an accident of directory layout.

## 2. Package Layout

```
training/
├── configs/          # base_config.py + one config per model, all hyperparameters live here
├── data/              # weighted_shard_sampler.py — the sync-tier × class-oversample weighting logic
├── models/            # (integration point) vendored DeepFilterNet3/CleanUMamba model source goes here
├── losses/            # (integration point) multi-res spectral loss, local-SNR loss implementations
├── schedulers/         # (integration point) cosine-with-warmup, matching config.ini's lr_cycle_decay
├── trainers/           # base_trainer.py + per-model concrete trainers
├── callbacks/          # (integration point) checkpoint pruning, early stopping, eval-metric logging
├── utils/
├── scripts/            # train_se_primary.py, train_se_escalation.py, train_se_crosscheck.py,
│                        #   train_classifier.py, train_aec.py — one entry point per model
├── checkpoints/         # output
└── runs/                 # logs
inference/
├── engines/             # per-model inference wrappers
├── runtime/              # ONNX/TensorRT export, the tier-routing logic (Model 4 → Model 1/2)
└── utils/
```

**Why `models/`, `losses/`, and `schedulers/` are empty integration points, not implementations:** DeepFilterNet3 and CleanUMamba are both real, published, pretrained architectures — reimplementing them from scratch here would mean training from zero instead of warm-starting from the actual public checkpoints every config in this design deliberately uses. The correct engineering move is vendoring the real source (`pip install deepfilternet` / the CleanUMamba repo) into these folders, not rewriting it. Config, data-sync, and training-loop scaffolding — the parts genuinely specific to this project — are fully built.

## 3. Per-Model Feature List

**Model 1 — `aegis-se-primary` (DeepFilterNet3, zero-lookahead):**
- Warm-started from `fal/DeepFilterNet3`, not trained from scratch
- `df_lookahead=0, conv_lookahead=0` — the one deliberate architecture change from stock, for the live-path latency budget
- LR = 5e-4 (half of stock's 1e-3, reflecting warm-start not from-scratch)
- Progressive batch schedule starting at 32 (vs. stock's 16) since early-training instability from-scratch doesn't apply here
- 50 epochs (vs. stock's 120) — fine-tune-scale, not pretraining-scale
- SNR distribution shifted toward the PS's literal >15dB target range
- Class-oversampling for the 4 confirmed-thin real-data classes, capped at 6× with an explicit stated ceiling

**Model 2 — `aegis-se-escalation` (DeepFilterNet3, stock lookahead, low-SNR-weighted):**
- Same warm-start, **no architecture change** — the lighter of the two DeepFilterNet3 fine-tunes
- LR = 2e-4, 25 epochs — smaller footprint reflecting a pure data-distribution shift, not a receptive-field change
- SNR sampling reweighted toward -5/0dB (30%/25%) per Shetu, Habets & Brendel's finding that low-SNR-weighted training improves *all* SNR conditions, not just hard ones

**Model 3 — `aegis-se-crosscheck` (CleanUMamba):**
- Uses the paper's own documented *post-pruning fine-tune* recipe (100K steps, batch 16, lr=2e-4), not its from-scratch recipe — extended to 150K given a more diverse data mix than the paper's original DNS-only training
- Hard-coded reminder in both the config docstring and the entry script's runtime output: evaluate at 48kHz on AEGIS's own shards, never cite the paper's 16kHz DNS-2020 numbers as this model's baseline — this was the exact bug caught in the earlier sync-tier review, now made structurally hard to repeat

**Model 4 — `aegis-clf-gate` (SNR/harmonic classifier):**
- Every hyperparameter explicitly marked as a reasoned default, not a citation — the one model in this design with no published reference to ground against
- 3-way gate taxonomy (`harmonic`/`impulsive`/`speech_dominant`) as an explicit crosswalk dict from the 10-class unified taxonomy, not implicit logic
- Inverse-frequency class-weighted loss, because impulsive events are far rarer in raw seconds than continuous noise even where real recording *counts* are high
- `validate_against_se_gap=True` — a named integration test verifying the classifier's routing decision actually correlates with where Model 1 measurably underperforms Model 2, not just a standalone accuracy number

**Model 5 — `aegis-aec-gate`:**
- `train_by_default=False` is a load-bearing flag the entry script actually checks and refuses to bypass without `--force` — verified in the smoke test
- Placeholder hyperparameters explicitly labeled as placeholders (no DeepVQE hyperparameter table exists to ground them against)

## 4. The One Piece of Shared Infrastructure Worth Explaining in Depth: `compute_sample_weight`

Every sample's training weight is the **product**, not a choice between, two independent corrections established across the data-forge review:
1. **Sync-tier weight** — down-weights Tier 3 (16kHz-native) sources so band-limited content doesn't teach the model a false "silence above 8kHz" signature for whichever classes happen to be Tier 3.
2. **Class-oversample factor** — compensates for real-hour scarcity on the four thin defence-specific classes, without inventing synthetic data (which this design explicitly removed).

A NOISEX-92 armored-vehicle clip is *both* Tier 3 *and* the thinnest class — it needs both corrections applied together, which is exactly what the verified smoke test proved (`0.25 × 6.0 = 1.5`, not a pick-one of the two).

## 5. Review & Finalize Checklist

- ✅ Every SE-branch model reads from the identical shard set — no repeat of the "compared two models trained on different data" bug from earlier in this review.
- ✅ Sync-tier and class-oversample weighting is defined once (`base_config.py`, `weighted_shard_sampler.py`), not duplicated per-model.
- ✅ Hyperparameters traced to primary sources where they exist (Models 1-3), explicitly marked as reasoned defaults where they don't (Model 4) or as placeholders needing a real literature pass (Model 5).
- ✅ CleanUMamba's cross-rate evaluation trap is now a structural reminder in the code itself, not just a design-doc footnote.
- ✅ Model 5's "don't train by accident" safety gate is real, tested code, not a comment.
- ✅ Config→dataset→weighting wiring smoke-tested end-to-end and passing (checkpoint naming, combined weight computation, all 5 configs loading with correct tuned values).
- ⚠️ **Not yet done, correctly scoped out of this pass:** the actual `models/`, `losses/`, `schedulers/` integration points need the real DeepFilterNet3/CleanUMamba source vendored in — this is a dependency-installation task on a machine with real network access, not a design gap.
- ⚠️ **Open, inherited from the data-forge review, unchanged by this pass:** rotor/helicopter and wind remain thin classes with no further data available — the oversampling in Model 1/2/3's configs mitigates under-exposure during training, it does not and cannot fix the underlying scarcity. Report those two classes' eval numbers with that caveat every time, per the standing disclosure requirement.

## 6. What Happens Next

1. Vendor real DeepFilterNet3 and CleanUMamba source into `training/models/`.
2. Implement the concrete `training_step`/`eval_step` methods in per-model trainer subclasses of `BaseTrainer`.
3. Run `python -m training.scripts.train_se_primary` for real, on a machine with `torch`+`webdataset` installed and the data-forge shards actually built.
4. Confirm the Section 7 evaluation protocol (from the earlier ML-architecture pass) runs against Model 1's output and reports per-class, not just aggregate, metrics.
