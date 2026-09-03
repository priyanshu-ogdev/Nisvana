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
from inference.runtime.audio_stream import StreamingAudioProcessor
from inference.runtime.escalation_router import AcousticEscalationRouter
from inference.utils.audio_io import load_audio_48k, save_audio_48k
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
    parser.add_argument("--use-hybrid-anc", action="store_true",
                        help="Enable secondary Normalized LMS adaptive filter stage.")
    parser.add_argument("--use-streaming", action="store_true", default=True,
                        help="Use overlap-add streaming processor.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional path to model checkpoint .pt file.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"=== PROJECT AEGIS: ENHANCING AUDIO FILE ===")
    print(f"Input file : {input_path}")
    print(f"Output file: {output_path}")
    print(f"Model      : {args.model}")
    print(f"Hybrid ANC : {args.use_hybrid_anc}")

    noisy_audio, sr = load_audio_48k(input_path, target_sr=48000)
    print(f"Loaded {len(noisy_audio) / sr:.2f} seconds of 48kHz audio ({len(noisy_audio)} samples)")

    t0 = time.perf_counter()

    if args.model == "router":
        router = AcousticEscalationRouter()
        processor = StreamingAudioProcessor(
            enhancement_fn=lambda x: router.route_and_enhance(x)[0],
            sample_rate=48000,
            frame_size=960,
            hop_size=480,
        )
        enhanced = processor.process_chunk(noisy_audio)
        flushed = processor.flush()
        if len(flushed) > 0:
            enhanced = np.concatenate([enhanced, flushed])
    else:
        model = build_model_for_key(args.model)
        model.eval()

        if args.checkpoint and Path(args.checkpoint).exists():
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"], strict=False)

        if args.use_hybrid_anc:
            pipeline = HybridAncPipeline(ai_model=model, enable_adaptive_filter=True)
            enhance_fn = lambda x: pipeline.process_frame(x)
        else:
            def enhance_fn(x):
                t_in = torch.from_numpy(x).float().unsqueeze(0)
                with torch.no_grad():
                    out = model(t_in)
                return out.squeeze().cpu().numpy()

        if args.use_streaming:
            processor = StreamingAudioProcessor(
                enhancement_fn=enhance_fn,
                sample_rate=48000,
                frame_size=960,
                hop_size=480,
            )
            enhanced = processor.process_chunk(noisy_audio)
            flushed = processor.flush()
            if len(flushed) > 0:
                enhanced = np.concatenate([enhanced, flushed])
        else:
            enhanced = enhance_fn(noisy_audio)

    t1 = time.perf_counter()
    duration_s = len(noisy_audio) / sr
    proc_time_s = t1 - t0
    rtf = proc_time_s / max(duration_s, 1e-6)

    # Save enhanced output
    saved_file = save_audio_48k(output_path, enhanced, sr=48000)
    print(f"\nSaved enhanced audio to: {saved_file}")
    print(f"Processing time: {proc_time_s:.3f} s (Real-Time Factor: {rtf:.3f}x)")

    # Perceptual quality indicators
    dns_in = compute_dnsmos_proxy(noisy_audio, sr=48000)
    dns_out = compute_dnsmos_proxy(enhanced, sr=48000)
    print(f"\n--- OBJECTIVE MOS ESTIMATION ---")
    print(f"  Input  DNSMOS (OVRL) : {dns_in['dnsmos_ovrl']:.2f} / 5.00")
    print(f"  Output DNSMOS (OVRL) : {dns_out['dnsmos_ovrl']:.2f} / 5.00")


if __name__ == "__main__":
    main()
