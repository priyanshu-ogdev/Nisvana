# Limiter Lookahead Deviation Memo

**Date:** 2026-09-10
**Author:** AI Agent
**Component:** `aegis-backend/src/audio/limiter.py`

## The Deviation
The sealed design (v16) specified a `1.0ms` lookahead for the output Peak Limiter.
The actual implemented lookahead is **`0.75ms`**.

## Justification
The overall system latency budget dictates a strict gate of `≤ 0.8ms` added latency for the final output limiter stage. A lookahead of 1.0ms inherently adds exactly 1.0ms of latency (as the buffer must be filled ahead of the gain calculation), failing the gate. 

Reducing the lookahead to `0.75ms` mathematically satisfies the `≤ 0.8ms` constraint while maintaining sufficient advance warning to duck high-energy transients (muzzle blasts).

## Evidence of Safety
The blast-ceiling test (`test_08_output_limiter_ceiling`) fires a simulated 155dB Friedlander blast (the acoustic profile of a 5.56mm muzzle blast at 1 meter) into the limiter.

The requirement is that the limiter must never allow the signal to exceed `-1dBFS` (`0.8912` linear).

**Test Output:**
```text
✅ test_08: Limiter holds ceiling 0.8912, latency 0.75ms
```

The ceiling holds. The 0.75ms lookahead is fast enough to react to the near-instantaneous onset of the shockwave without clipping the output or violating the total latency budget.
