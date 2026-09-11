"""
inference/scripts/export_onnx.py — Edge Hardware ONNX Export CLI
"""

import argparse
from pathlib import Path
import torch

from data_forge.config import DATA_DIR
from training.models.model_loader import build_model_for_key
from inference.engines.onnx_engine import (
    export_to_onnx,
    export_platform_b_tensorrt,
    benchmark_edge_latency,
    get_algorithmic_delay_ms,
)
from inference.contract import DEFAULT_STREAMING_CONTRACT


def main():
    parser = argparse.ArgumentParser(description="Export Project AEGIS models to ONNX for Edge / Jetson / Blackwell")
    parser.add_argument("--model", type=str, default="aegis-se-primary",
                        choices=["aegis-se-primary", "aegis-se-escalation", "aegis-se-crosscheck", "aegis-clf-gate"],
                        help="Model key to export.")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR / "onnx_models"),
                        help="Directory to save exported ONNX model.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_STREAMING_CONTRACT.sample_rate,
                        help="Sample rate in Hz.")
    parser.add_argument("--chunk-ms", type=float, default=DEFAULT_STREAMING_CONTRACT.chunk_ms,
                        help="Frame chunk duration in milliseconds.")
    parser.add_argument("--platform", type=str, default="platform_a",
                        choices=["platform_a", "platform_b"],
                        help="Deployment target: platform_a (ARM/CPU QNNPACK) or platform_b (NVIDIA GPU/TensorRT).")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Export in FP16 precision (Platform B / TensorRT).")
    parser.add_argument("--profile", action="store_true", default=True,
                        help="Profile edge latency after export.")
    args = parser.parse_args()
    if args.sample_rate != DEFAULT_STREAMING_CONTRACT.sample_rate:
        parser.error(
            f"--sample-rate must be {DEFAULT_STREAMING_CONTRACT.sample_rate} "
            "to match the streaming contract"
        )
    if abs(args.chunk_ms - DEFAULT_STREAMING_CONTRACT.chunk_ms) > 1e-6:
        parser.error(
            f"--chunk-ms must be {DEFAULT_STREAMING_CONTRACT.chunk_ms:g} "
            "to match the streaming contract"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.onnx"

    print(f"=== EXPORTING {args.model} TO ONNX ({args.platform.upper()}) ===")
    model = build_model_for_key(args.model)
    model.eval()

    if args.platform == "platform_b":
        precision = "fp16" if args.fp16 else "fp32"
        dest = export_platform_b_tensorrt(
            model=model,
            output_path=out_path,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
            precision=precision,
        )
    else:
        dest = export_to_onnx(
            model=model,
            output_path=out_path,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
        )
    print(f"Successfully exported ONNX model to: {dest} ({dest.stat().st_size / (1024*1024):.2f} MB)")

    if args.profile:
        print(f"\n--- PROFILING EDGE LATENCY (chunk={args.chunk_ms} ms) ---")
        delay_ms = get_algorithmic_delay_ms(args.model, model=model)
        metrics = benchmark_edge_latency(
            model=model,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
            algorithmic_delay_ms=delay_ms,
            num_runs=30,
        )
        for k, v in metrics.items():
            print(f"  {k:<25}: {v}")


if __name__ == "__main__":
    main()
