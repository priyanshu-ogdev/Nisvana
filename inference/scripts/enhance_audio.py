"""
inference/scripts/enhance_audio.py — Offline Audio File Enhancement CLI

Enhances noisy WAV/FLAC files using:
- Model 1 (aegis-se-primary, 0ms lookahead)
- Model 2 (aegis-se-escalation, 40ms lookahead)
- Model 3 (aegis-se-crosscheck, CleanUMamba)
- Hybrid AI + NLMS Adaptive Filter pipeline
- Dynamic Acoustic Escalation Router
"""

import argparse
from pathlib import Path
import time
import numpy as np
import torch

from training.models.model_loader import build_model_for_key
from inference.runtime.hybrid_anc import HybridAncPipeline
from inference.runtime.audio_stream import StatefulHopProcessor
from inference.runtime.escalation_router import AcousticEscalationRouter
from inference.engines.onnx_model_adapter import OnnxModelAdapter, build_onnx_backed_router
from inference.utils.audio_io import load_audio_48k, save_audio_48k
from inference.utils.sih_metrics import evaluate_sih_compliance
from training.utils.metrics import (
    compute_snr_db,
    compute_stoi,
    compute_pesq_proxy,
    compute_dnsmos_proxy,
)


def main():
    parser = argparse.ArgumentParser(description="Enhance noisy audio file with Project AEGIS models")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to noisy input audio file.")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Path to destination enhanced audio file.")
    parser.add_argument("--model", "-m", type=str, default="aegis-se-primary",
                        choices=["aegis-se-primary", "aegis-se-escalation", "aegis-se-crosscheck", "router"],
                        help="Model key to use or 'router' for dynamic acoustic escalation.")
    parser.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch",
                        help="Inference backend. ONNX requires exported model files.")
    parser.add_argument("--onnx-dir", type=str, default=None,
                        help="Directory containing <model-key>.onnx files.")
    parser.add_argument("--onnx-provider", action="append", default=None,
                        help="ONNX execution provider; repeat to set priority order.")
    parser.add_argument("--use-hybrid-anc", action="store_true",
                        help="Enable secondary Normalized LMS adaptive filter stage.")
    parser.add_argument("--use-streaming", action=argparse.BooleanOptionalAction, default=True,
                        help="Use chunked streaming processing (StatefulHopProcessor -- "
                             "hop-synchronous, no windowing, correct for this project's "
                             "stateful causal models; was mislabeled 'overlap-add' before "
                             "this pass's fix replaced the OLA processor it used to wrap).")
    parser.add_argument("--reference-clean", "-c", type=str, default=None,
                        help="Optional path to clean ground-truth audio to compute SIH Benchmark Scorecard.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional path to model checkpoint .pt file.")
    parser.add_argument("--prepare-for-transmission", action="store_true",
                        help="Additionally save a narrowband, radio-ready version of the enhanced "
                             "output (downsampled + pre-emphasized + dithered) -- without this, "
                             "only the internal-rate 48kHz file is produced, which is not what a "
                             "real tactical radio channel would actually carry.")
    parser.add_argument("--transmission-sample-rate", type=int, default=16000,
                        help="Target narrowband rate for --prepare-for-transmission (default: 16000 Hz).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"=== PROJECT AEGIS: ENHANCING AUDIO FILE ===")
    print(f"Input file : {input_path}")
    print(f"Output file: {output_path}")
    print(f"Model      : {args.model}")
    print(f"Hybrid ANC : {args.use_hybrid_anc}")
    print(f"Backend    : {args.backend}")

    if args.backend == "onnx" and args.onnx_dir is None:
        parser.error("--onnx-dir is required with --backend onnx")
    if args.backend == "onnx" and args.checkpoint:
        parser.error("--checkpoint applies only to the PyTorch backend")
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else None

    def onnx_path(model_key: str) -> Path:
        path = onnx_dir / f"{model_key}.onnx"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing ONNX model for {model_key}: {path}. "
                "Export it before starting inference."
            )
        return path

    noisy_audio, sr = load_audio_48k(input_path, target_sr=48000)
    print(f"Loaded {len(noisy_audio) / sr:.2f} seconds of 48kHz audio ({len(noisy_audio)} samples)")

    t0 = time.perf_counter()

    if args.model == "router":
        if args.backend == "onnx":
            router = build_onnx_backed_router(
                onnx_path("aegis-se-primary"),
                onnx_path("aegis-se-escalation"),
                onnx_path("aegis-clf-gate"),
                providers=args.onnx_provider,
            )
        else:
            router = AcousticEscalationRouter()
        # MERGE-PASS FIX: was StreamingAudioProcessor (OLA, 50% overlap,
        # frame_size=960/hop_size=480) wrapping this router -- wrong
        # pattern for a stateful, causal model chain (see
        # StatefulHopProcessor's docstring in audio_stream.py). Fixed to
        # the correct one-hop-in/one-hop-out pattern, with reset_fn wired
        # to the router's own reset_state so a fresh file/session starts
        # both models with clean hidden state.
        processor = StatefulHopProcessor(
            enhancement_fn=lambda x: router.route_and_enhance(x)[0],
            sample_rate=48000,
            hop_size=480,
            reset_fn=router.reset_state,
        )
        processor.reset()
        enhanced = processor.process_chunk(noisy_audio)
        flushed = processor.flush()
        if len(flushed) > 0:
            enhanced = np.concatenate([enhanced, flushed])
    else:
        if args.backend == "onnx":
            model = OnnxModelAdapter(onnx_path(args.model), providers=args.onnx_provider)
        else:
            model = build_model_for_key(args.model)
            model.eval()

        if args.checkpoint and Path(args.checkpoint).exists():
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"], strict=False)

        reset_fn = model.reset_state if hasattr(model, "reset_state") else None

        if args.use_hybrid_anc:
            pipeline = HybridAncPipeline(ai_model=model, enable_adaptive_filter=True)
            enhance_fn = lambda x: pipeline.process_frame(x)
            reset_fn = pipeline.reset_state
        else:
            def enhance_fn(x):
                t_in = torch.from_numpy(x).float().unsqueeze(0)
                with torch.no_grad():
                    out = model(t_in)
                return out.squeeze().cpu().numpy()

        if args.use_streaming:
            # MERGE-PASS FIX: same OLA-vs-stateful-model issue as the
            # router branch above.
            processor = StatefulHopProcessor(
                enhancement_fn=enhance_fn,
                sample_rate=48000,
                hop_size=480,
                reset_fn=reset_fn,
            )
            processor.reset()
            enhanced = processor.process_chunk(noisy_audio)
            flushed = processor.flush()
            if len(flushed) > 0:
                enhanced = np.concatenate([enhanced, flushed])
        else:
            if reset_fn is not None:
                reset_fn()
            enhanced = enhance_fn(noisy_audio)

    t1 = time.perf_counter()
    duration_s = len(noisy_audio) / sr
    proc_time_s = t1 - t0
    rtf = proc_time_s / max(duration_s, 1e-6)

    # Save enhanced output
    saved_file = save_audio_48k(output_path, enhanced, sr=48000)
    print(f"\nSaved enhanced audio to: {saved_file}")
    print(f"Processing time: {proc_time_s:.3f} s (Real-Time Factor: {rtf:.3f}x)")

    if args.prepare_for_transmission:
        from inference.utils.transmission_prep import prepare_for_transmission, TransmissionPrepConfig

        tx_config = TransmissionPrepConfig(target_sample_rate=args.transmission_sample_rate)
        tx_audio, tx_sr = prepare_for_transmission(enhanced, source_sample_rate=48000, config=tx_config)
        tx_path = output_path.with_name(output_path.stem + f"_tx_{tx_sr}hz" + output_path.suffix)
        save_audio_48k(tx_path, tx_audio, sr=tx_sr, dither=False)  # already dithered by prepare_for_transmission
        print(f"Saved transmission-ready ({tx_sr}Hz, narrowband) audio to: {tx_path}")

    # Perceptual quality indicators
    dns_in = compute_dnsmos_proxy(noisy_audio, sr=48000)
    dns_out = compute_dnsmos_proxy(enhanced, sr=48000)
    print(f"\n--- OBJECTIVE MOS ESTIMATION ---")
    print(f"  Input  DNSMOS (OVRL) : {dns_in['dnsmos_ovrl']:.2f} / 5.00")
    print(f"  Output DNSMOS (OVRL) : {dns_out['dnsmos_ovrl']:.2f} / 5.00")

    # Evaluate against official SIH benchmarks if reference clean audio is provided
    if args.reference_clean and Path(args.reference_clean).exists():
        clean_ref, _ = load_audio_48k(args.reference_clean, target_sr=48000)
        # Trim to matching length
        min_len = min(len(clean_ref), len(enhanced), len(noisy_audio))
        
        from inference.engines.onnx_engine import get_algorithmic_delay_ms

        # FIX (this pass): two real issues here, not one.
        # (a) This directly read the static MODEL_ALGORITHMIC_DELAY_MS dict,
        #     bypassing get_algorithmic_delay_ms's fallback-vs-vendored
        #     reconciliation entirely -- the exact bug that function was
        #     built to fix, still live at this actual call site.
        # (b) Separately, and pre-existing: when args.model == "router",
        #     `model` is never defined (only `router` is, see the branch
        #     above) -- "router" was also never a real key in the delay
        #     dict, so .get("router", 0.0) was ALWAYS silently returning
        #     the 0.0 default, regardless of which internal model (primary/
        #     escalation, fallback/vendored) the router actually used for
        #     a given run. A router mixes modes per-chunk, so there's no
        #     single exact figure to report here -- using model_primary's
        #     as a stated approximation is more honest than the previous
        #     silent, unexplained 0.0.
        if args.model == "router":
            delay_source_model = router.model_primary
            delay_lookup_key = "aegis-se-primary"
            print("  Note: 'router' mode mixes primary/escalation models per-chunk; "
                  "algorithmic delay below approximates using the primary model's "
                  "figure, not an exact per-chunk value.")
        else:
            delay_source_model = model
            delay_lookup_key = args.model

        algorithmic_delay_ms = get_algorithmic_delay_ms(delay_lookup_key, model=delay_source_model)
        compute_latency_ms = rtf * 10.0
        total_latency_ms = compute_latency_ms + algorithmic_delay_ms
        
        sih_res = evaluate_sih_compliance(
            estimate=enhanced[:min_len],
            target_clean=clean_ref[:min_len],
            input_noisy=noisy_audio[:min_len],
            sample_rate=48000,
            total_latency_ms=total_latency_ms,
            chunk_ms=10.0,
        )
        print("\n" + sih_res.format_markdown_table("OFFICIAL SIH DEFENCE BENCHMARK SCORECARD"))


if __name__ == "__main__":
    main()
