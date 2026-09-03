"""
inference/scripts/export_onnx.py — Edge Hardware ONNX Export CLI
"""

import argparse
from pathlib import Path
import torch

from data_forge.config import DATA_DIR
from training.models.model_loader import build_model_for_key
from inference.engines.onnx_engine import export_to_onnx, benchmark_edge_latency


def main():
    parser = argparse.ArgumentParser(description="Export Project AEGIS models to ONNX for Edge / Jetson")
    parser.add_argument("--model", type=str, default="aegis-se-primary",
                        choices=["aegis-se-primary", "aegis-se-escalation", "aegis-se-crosscheck", "aegis-clf-gate"],
                        help="Model key to export.")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR / "onnx_models"),
                        help="Directory to save exported ONNX model.")
    parser.add_argument("--sample-rate", type=int, default=48000,
                        help="Sample rate in Hz.")
    parser.add_argument("--chunk-ms", type=float, default=10.0,
                        help="Frame chunk duration in milliseconds.")
    parser.add_argument("--profile", action="store_true", default=True,
                        help="Profile edge latency after export.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.onnx"

    print(f"=== EXPORTING {args.model} TO ONNX ===")
    model = build_model_for_key(args.model)
    model.eval()

    dest = export_to_onnx(
        model=model,
        output_path=out_path,
        sample_rate=args.sample_rate,
        chunk_ms=args.chunk_ms,
    )
    print(f"Successfully exported ONNX model to: {dest} ({dest.stat().st_size / (1024*1024):.2f} MB)")

    if args.profile:
        print(f"\n--- PROFILING EDGE LATENCY (chunk={args.chunk_ms} ms) ---")
        metrics = benchmark_edge_latency(
            model=model,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
            num_runs=30,
        )
        for k, v in metrics.items():
            print(f"  {k:<20}: {v}")


if __name__ == "__main__":
    main()
