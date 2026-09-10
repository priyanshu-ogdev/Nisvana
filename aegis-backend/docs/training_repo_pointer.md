# Training Repository Pointer

**Date:** 2026-09-10
**Status:** Verification Required

Per the v16 sealed design (and v13§2a/2b), the model training repository MUST contain specific synthetic data generators that are NOT part of this edge inference backend.

## Required Generators in the Training Repo:
1. **`friedlander.py`**: A synthesizer generating Friedlander waves representing muzzle blasts (140-160dB, 1-18ms duration, <3-7ms positive phase). 
2. **`nwave.py`**: A synthesizer generating N-waves representing supersonic bullet bow shockwaves.
3. **Hard-ADC-saturation clipping**: Data augmentation steps simulating pre-ADC microphone clipping.

## Instructions for Data Team
Do **not** substitute these with generic clicks or impulse noises. The models (DeepFilterNet3 and CleanUMamba) depend on the unique spectral and temporal signatures of shockwaves to correctly reject them without muting subsequent speech.

If these files are absent from the main training/forge repo, an issue must be opened immediately.

## Open Acoustic Gaps
Sirens and wind remain explicitly labeled as **open gaps** per v16. We do not silently claim they are solved in the training data until validated.
