#!/usr/bin/env bash
# burn_test_loop.sh — 4-hour stability burn test
# Exits 0 only if all assertions hold throughout the run.
# Run on Pi: bash scripts/burn_test_loop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
DURATION_SECS=${1:-14400}  # default: 4 hours

echo "=== PROJECT AEGIS BURN TEST ==="
echo "Duration: ${DURATION_SECS}s"
echo "Start: $(date)"

source "$ROOT/venv/bin/activate"
cd "$ROOT"

FAIL=0
FRAME_COUNT=0
NAN_COUNT=0

python3 - <<PYEOF
import sys, time, numpy as np
sys.path.insert(0, '.')
from src.audio.limiter import PeakLimiter, CEILING_LINEAR
from src.audio.rnnoise_vad import RNNoiseVAD
from src.dsp.fft_exporter import FftExporter

SAMPLE_RATE = 48000
FRAME_SIZE = 480
DURATION = ${DURATION_SECS}
frames_per_sec = SAMPLE_RATE / FRAME_SIZE

limiter = PeakLimiter(SAMPLE_RATE)
vad = RNNoiseVAD(SAMPLE_RATE, FRAME_SIZE)
fft = FftExporter(SAMPLE_RATE)

start = time.monotonic()
frame_count = 0
nan_count = 0
ceiling_violations = 0

print("Running burn loop...")
while time.monotonic() - start < DURATION:
    t = time.monotonic()
    frame = np.random.randn(FRAME_SIZE).astype(np.float32) * 0.5
    # Occasionally inject blast
    if frame_count % 300 == 0:
        frame *= 200.0

    limited = limiter.process_frame(frame)
    
    # Check NaN
    if np.any(np.isnan(limited)):
        nan_count += 1
    
    # Check ceiling
    max_out = float(np.max(np.abs(limited)))
    if max_out > CEILING_LINEAR + 1e-4:
        ceiling_violations += 1
    
    # VAD
    vad.process(frame)
    
    # FFT (30fps rate)
    if frame_count % int(frames_per_sec / 30) == 0:
        bins = fft.compute_bins(limited)
        if len(bins) != 64:
            print(f"ERROR: FFT returned {len(bins)} bins (expected 64)")
            sys.exit(1)

    frame_count += 1
    time.sleep(max(0, FRAME_SIZE / SAMPLE_RATE - (time.monotonic() - t)))

print(f"Frames processed: {frame_count}")
print(f"NaN outputs: {nan_count}")
print(f"Ceiling violations: {ceiling_violations}")

if nan_count > 0 or ceiling_violations > 0:
    print("BURN TEST: FAIL")
    sys.exit(1)
else:
    print("BURN TEST: PASS")
    sys.exit(0)
PYEOF

echo "End: $(date)"
echo "=== BURN TEST COMPLETE ==="
