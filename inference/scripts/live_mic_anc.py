"""
inference/scripts/live_mic_anc.py — Real-Time Live Microphone & Headset ANC Prototype

Demonstrates live low-latency speech enhancement and active noise cancellation:
- Streams 48,000 Hz audio in real-time chunks (10 ms = 480 samples)
- Simulates the project's actual hardware input: an N-mic air array PLUS a
  throat-contact mic, fused via MultichannelHardwareFrontend into the mono
  stream the model pipeline expects (previously this script simulated a
  single mic signal with no trace of the array/throat-mic hardware design
  anywhere -- fixed this pass, see run_live_simulation's docstring)
- Integrates Hybrid AI (DeepFilterNet3) + Normalized LMS adaptive filter
- Dynamically escalates based on Model 4 acoustic environment gating
- Paced in real-time to simulate embedded soldier comms headset operation
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
import torch

from training.models.model_loader import build_model_for_key
from inference.runtime.hybrid_anc import HybridAncPipeline
from inference.runtime.escalation_router import AcousticEscalationRouter
from inference.runtime.audio_stream import StatefulHopProcessor
from inference.runtime.multichannel_frontend import MultichannelHardwareFrontend, HardwareFrontendConfig
from inference.engines.onnx_model_adapter import OnnxModelAdapter


def run_live_simulation(duration_seconds: float = 5.0, chunk_ms: float = 10.0, num_air_mics: int = 4,
                         save_output_path: Optional[str] = None, backend: str = "pytorch",
                         onnx_dir: Optional[str] = None, onnx_providers: Optional[list] = None):
    """Simulates real-time tactical headset streaming with dynamic acoustic conditions.

    FIX (multichannel hardware, carried from an earlier pass): previously
    synthesized a single `primary_mic` signal directly, with no trace of
    this project's own hardware design (a multi-mic array plus a
    throat-contact mic) anywhere in the live simulation path. Now
    synthesizes `num_air_mics` air channels (each with independent
    mic-placement noise, so they're not identical copies) plus a
    throat-mic channel, and routes them through MultichannelHardwareFrontend
    before anything else touches the audio.

    MERGE-PASS FIX (this pass): the pipeline construction and processing
    loop below had NOT received an earlier, separate inference-layer fix
    for a serious bug -- confirmed absent from this lineage by diff against
    that review's output. The bug: this script constructed
    AcousticEscalationRouter, called `router.route_and_enhance()`, and
    discarded its ACTUAL enhanced-audio output (kept only status metadata
    for the console printout), while the real output audio came from a
    SEPARATE HybridAncPipeline fixed to `ai_model=model_primary` --
    meaning the escalation ladder had zero effect on the actual output
    regardless of what the router decided or what the console printed.
    Also, that separate pipeline was wrapped in StreamingAudioProcessor
    (Hann-window overlap-add), the wrong pattern for this project's
    stateful causal models (see StatefulHopProcessor's docstring in
    audio_stream.py). Both fixed together here: the pipeline is now
    router-backed (one call, real routing decision reaches the output),
    and streaming uses StatefulHopProcessor (hop-synchronous, no
    windowing, correct for stateful models).
    """
    print("========================================================================")
    print("PROJECT AEGIS — REAL-TIME LIVE ANC PROTOTYPE DEMONSTRATION")
    print(f"Configuration: 48,000 Hz, {chunk_ms} ms chunk size (480 samples), Hybrid AI+NLMS")
    print(f"Hardware input: {num_air_mics}-mic air array + throat-contact mic -> MultichannelHardwareFrontend")
    print("========================================================================")

    if backend == "onnx":
        if onnx_dir is None:
            raise ValueError("onnx_dir is required when backend='onnx'")
        export_dir = Path(onnx_dir)

        def exported(key: str) -> OnnxModelAdapter:
            path = export_dir / f"{key}.onnx"
            if not path.exists():
                raise FileNotFoundError(f"Missing ONNX export: {path}")
            return OnnxModelAdapter(path, providers=onnx_providers)

        model_primary = exported("aegis-se-primary")
        model_escalation = exported("aegis-se-escalation")
        classifier = exported("aegis-clf-gate")
    elif backend == "pytorch":
        model_primary = build_model_for_key("aegis-se-primary")
        model_escalation = build_model_for_key("aegis-se-escalation")
        classifier = build_model_for_key("aegis-clf-gate")
    else:
        raise ValueError(f"Unsupported inference backend: {backend}")

    router = AcousticEscalationRouter(
        model_primary=model_primary,
        model_escalation=model_escalation,
        classifier=classifier,
    )
    # MERGE-PASS FIX: router-backed, not a separate ai_model=model_primary
    # pipeline -- see this function's docstring for why that separation
    # was the actual bug.
    hybrid_pipeline = HybridAncPipeline(router=router, enable_adaptive_filter=True)
    hardware_frontend = MultichannelHardwareFrontend(
        HardwareFrontendConfig(num_air_mics=num_air_mics, has_throat_mic=True)
    )

    sr = 48000
    chunk_samples = int(sr * chunk_ms / 1000.0)
    total_chunks = int(duration_seconds / (chunk_ms / 1000.0))

    # MERGE-PASS FIX: StatefulHopProcessor, not StreamingAudioProcessor --
    # one hop in, one hop out, no windowing, correct for this stateful
    # pipeline. reset_fn wired to the pipeline's own reset_state so
    # starting the stream also clears both models' hidden state (and, via
    # the router's reset_state, the classifier's context buffer and
    # crossfade tracking) in one call.
    processor = StatefulHopProcessor(
        enhancement_fn=lambda x: hybrid_pipeline.process_frame(x),
        sample_rate=sr,
        hop_size=chunk_samples,
        reset_fn=hybrid_pipeline.reset_state,
    )
    processor.reset()
    hardware_frontend.reset_state()

    print(f"\nStreaming started ({duration_seconds:.1f}s demo)...")
    print(f"{'Time (s)':<10} | {'Mode':<12} | {'Acoustic Class':<18} | {'Latency (ms)':<14} | {'Status'}")
    print("-" * 70)

    start_time = time.perf_counter()
    latencies = []
    # FIX (found during final audit): enhanced_out was computed every chunk
    # and only ever used for its processing TIME (proc_ms below) -- the
    # actual enhanced audio content was silently discarded, meaning this
    # script could confirm real-time TIMING but never let anyone actually
    # verify enhancement QUALITY, live or after the fact. Accumulated here
    # and optionally saved via --save-output so the demo produces an
    # actual artifact, not just a console timing report.
    output_chunks = []
    last_route_meta = {"mode": "primary", "category": "speech_dominant"}

    for i in range(total_chunks):
        t_chunk_start = time.perf_counter()
        elapsed_s = i * (chunk_ms / 1000.0)

        # Generate synthetic live audio representing dynamic mission conditions
        t = np.linspace(elapsed_s, elapsed_s + chunk_ms / 1000.0, chunk_samples, dtype=np.float32)
        speech = (np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 900 * t)) * 0.1

        # Simulate rotating defence noise scenarios across time
        if elapsed_s < 1.5:
            # Rotor / Drone harmonic noise
            noise = np.sin(2 * np.pi * 150 * t) * 0.25 + np.sin(2 * np.pi * 300 * t) * 0.15
        elif elapsed_s < 3.0:
            # Gunfire / artillery blast impulse
            noise = np.random.randn(chunk_samples).astype(np.float32) * 1.5 if (i % 20 < 5) else np.zeros(chunk_samples, dtype=np.float32)
        else:
            # Low-frequency tank diesel rumble
            noise = np.sin(2 * np.pi * 65 * t) * 0.35

        # Simulate the REAL hardware capture: N air mics at different
        # positions on the headset each pick up a slightly different
        # mix of speech+noise (occlusion, distance-to-mouth, and
        # orientation-to-noise-source all differ per mic in reality) --
        # not N identical copies of one signal, which would defeat the
        # whole point of the energy-weighted array combination.
        air_mics = np.stack([
            speech * (0.85 + 0.15 * np.random.rand()) + noise * (0.7 + 0.6 * np.random.rand())
            for _ in range(num_air_mics)
        ]).astype(np.float32)

        # Throat mic: picks up voiced speech via bone conduction, largely
        # immune to the airborne noise term entirely -- reflected here by
        # NOT including `noise` in its signal at all, only a much smaller
        # coupling term plus its own sensor noise floor.
        throat_mic = (speech * 1.1 + np.random.randn(chunk_samples).astype(np.float32) * 0.01).astype(np.float32)

        primary_mic = hardware_frontend.process(air_mics, throat_mic_channel=throat_mic)

        # MERGE-PASS FIX: single call now produces both the real enhanced
        # audio AND the routing metadata used for the status line below --
        # previously these were two separate calls, and only the audio
        # from the WRONG one (always-primary, via StreamingAudioProcessor)
        # was actually used.
        t0 = time.perf_counter()
        enhanced_out = processor.process_chunk(primary_mic)
        output_chunks.append(enhanced_out.copy())
        t1 = time.perf_counter()

        last_route_meta = router.last_prediction

        proc_ms = (t1 - t0) * 1000.0
        latencies.append(proc_ms)

        # Log status every 500 ms
        if i % 50 == 0 or i == total_chunks - 1:
            print(f"{elapsed_s:<10.2f} | {last_route_meta['mode']:<12} | {last_route_meta['category']:<18} | {proc_ms:<14.2f} | PASS (Real-Time)")

        # Pace in real-time
        chunk_elapsed = time.perf_counter() - t_chunk_start
        sleep_time = max(0.0, (chunk_ms / 1000.0) - chunk_elapsed)
        time.sleep(sleep_time)

    total_elapsed = time.perf_counter() - start_time
    mean_lat = float(np.mean(latencies))
    rtf = mean_lat / chunk_ms

    print("\n========================================================================")
    print("LIVE STREAMING PROTOTYPE COMPLETED")
    print(f"Total stream time : {total_elapsed:.2f} s ({total_chunks} chunks)")
    print(f"Mean frame latency: {mean_lat:.2f} ms per {chunk_ms} ms chunk")
    print(f"Real-Time Factor  : {rtf:.3f}x (< 1.0 indicates full real-time readiness)")
    print("========================================================================")

    if save_output_path:
        full_output = np.concatenate(output_chunks) if output_chunks else np.array([], dtype=np.float32)
        out_path = Path(save_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, full_output, 48000, subtype="PCM_16")
        print(f"Enhanced audio saved to: {out_path} ({len(full_output) / 48000:.2f}s) -- listen to actually verify quality, not just timing.")


def main():
    parser = argparse.ArgumentParser(description="Live Microphone & Headset ANC Prototype Demonstration")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Duration of live streaming session in seconds.")
    parser.add_argument("--chunk-ms", type=float, default=10.0,
                        help="Audio frame chunk size in milliseconds (default: 10ms).")
    parser.add_argument("--num-air-mics", type=int, default=4,
                        help="Number of air-conduction mics in the array (matches this project's hardware spec).")
    parser.add_argument("--save-output", type=str, default=None,
                        help="Path to save the enhanced audio as a WAV file after the run -- without this, "
                             "the demo only reports timing, with no way to actually verify enhancement quality.")
    parser.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch")
    parser.add_argument("--onnx-dir", type=str, default=None,
                        help="Directory containing <model-key>.onnx exports.")
    parser.add_argument("--onnx-provider", action="append", default=None,
                        help="ONNX execution provider; repeat to set priority order.")
    args = parser.parse_args()

    run_live_simulation(
        duration_seconds=args.duration,
        chunk_ms=args.chunk_ms,
        num_air_mics=args.num_air_mics,
        save_output_path=args.save_output,
        backend=args.backend,
        onnx_dir=args.onnx_dir,
        onnx_providers=args.onnx_provider,
    )


if __name__ == "__main__":
    main()
