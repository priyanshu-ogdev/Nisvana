"""
Project AEGIS — Unified CLI Interface
Supports operations for both 4TB server deployment and local test verification:
  python -m data_forge fetch [--source <name>|all] [--sample-mode] [--server-mode] [--dry-run]
  python -m data_forge preprocess [--max-workers <N>] [--commercial-strict]
  python -m data_forge augment
  python -m data_forge mix [--num-mixtures <N>] [--min-snr <float>] [--max-snr <float>]
  python -m data_forge verify
  python -m data_forge export [--card-only]
  python -m data_forge run-all [--sample-mode] [--server-mode] [--dry-run]
"""

import argparse
import sys
from pathlib import Path
from data_forge.bibliography import export_bibliography_json
from data_forge.config import (
    AUGMENTED_DIR,
    BRANCH_SE,
    DATA_DIR,
    ForgeMixingConfig,
    MANIFESTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
)
from data_forge.fetcher import FetchManager
from data_forge.preprocessor import LicenseComplianceMode, PreprocessingPipeline
from data_forge.augmentor import AugmentationEngine
from data_forge.mixer import AecBranch, ClassifierBranch, SpeechEnhancementBranch
from data_forge.verifier import AuditReporter, PipelineAuditor
from data_forge.exporter import pack_branch_to_shards, write_dataset_card
from data_forge.config import BRANCH_AEC, BRANCH_CLASSIFIER, SAMPLES_PER_SHARD, SHARDS_DIR


def cmd_fetch(args):
    print(">>> Executing Data Fetcher...")
    manager = FetchManager(RAW_DIR)
    is_full = getattr(args, "full_mode", False) or getattr(args, "server_mode", False)
    sample_mode = args.sample_mode and not is_full

    if args.source and args.source != "all":
        results = manager.fetch_source(args.source, sample_mode=sample_mode, dry_run=args.dry_run)
    else:
        results = manager.fetch_all(sample_mode=sample_mode, dry_run=args.dry_run)

    print(">>> Fetch completed.")


def cmd_preprocess(args):
    print(">>> Executing 10-Step Preprocessing Pipeline...")
    lic_mode = LicenseComplianceMode.COMMERCIAL_STRICT if args.commercial_strict else LicenseComplianceMode.RESEARCH_PROTOTYPE
    pipeline = PreprocessingPipeline(license_mode=lic_mode)
    res = pipeline.run_pipeline(max_workers=args.max_workers)
    print(f">>> Preprocessing completed. Processed: {res['processed_count']}, Rejected: {res['rejected_count']}")


def cmd_augment(args):
    print(">>> Executing Grounded Per-Source Augmentation...")
    engine = AugmentationEngine()
    counts = engine.run_augmentation()
    print(f">>> Augmentation completed. Generated: {counts}")


def cmd_mix(args):
    print(">>> Executing Data-Forge Multi-Branch Mixing...")
    cfg = ForgeMixingConfig(
        min_snr_db=args.min_snr,
        max_snr_db=args.max_snr,
    )

    # 1. Branch SE (Models 1-3)
    se_branch = SpeechEnhancementBranch(config=cfg)
    clean_speech_files = list((PROCESSED_DIR / "clean_speech").glob("*.wav"))
    noise_files = [
        f for f in PROCESSED_DIR.glob("**/*.wav")
        if "clean_speech" not in str(f)
        and "rir" not in str(f)
        and "aec" not in str(f)
        and "far_end_echo" not in str(f)
    ]
    # Include augmented noise files (excluding any speech or aec)
    augmented_noise = [
        f for f in AUGMENTED_DIR.glob("**/*.wav")
        if "clean_speech" not in str(f) and "aec" not in str(f)
    ]
    all_noise = noise_files + augmented_noise
    rir_files = list((PROCESSED_DIR / "rir").glob("*.wav")) + list((RAW_DIR / "openslr_rirs" / "rir_wavs").glob("*.wav"))

    if not clean_speech_files:
        print("Warning: No clean speech files found in data/processed/clean_speech.")
        return

    mixtures = se_branch.generate_mixtures(
        clean_files=clean_speech_files,
        noise_files=all_noise if all_noise else clean_speech_files,
        rir_files=rir_files,
        num_mixtures=args.num_mixtures,
    )

    # 2. Branch Classifier (Model 4)
    clf_branch = ClassifierBranch()
    noise_map = {f.name: f.parent.name for f in all_noise}
    clf_branch.build_dataset_from_mixtures(mixtures, noise_class_map=noise_map)

    # 3. Branch AEC (Model 5)
    aec_branch = AecBranch()
    aec_branch.organize_aec_pairs(RAW_DIR / "aec_challenge")

    print(">>> Multi-branch mixing completed.")


def cmd_verify(args):
    print(">>> Executing Pipeline Audit and Verification...")
    auditor = PipelineAuditor(DATA_DIR)
    summary = auditor.run_full_audit()
    report = AuditReporter.generate_markdown_report(summary)
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        safe_report = report.encode("ascii", errors="replace").decode("ascii")
        print("\n" + safe_report)


def cmd_export(args):
    """
    Upgrades data/forge/'s flat-file output into industry-standard
    WebDataset tar-shards (sequential-read, DataLoader-ready) plus an
    auto-generated dataset card. This is what a real training run should
    point at — not the loose files in data/forge/ directly.
    """
    print(">>> Exporting to WebDataset shards (industry-standard training format)...")
    args.card_only = getattr(args, "card_only", False)
    branch_stats = {}

    def se_groups(sample_root: Path):
        key = sample_root.name
        branch = sample_root.parent
        noisy = branch / "noisy" / f"{key}_noisy.wav"
        if not noisy.exists():
            noisy = branch / f"{key}_noisy.wav"
        clean = branch / "clean" / f"{key}_clean.wav"
        if not clean.exists():
            clean = branch / f"{key}_clean.wav"
        rir = branch / "rir" / f"{key}_rir.wav"
        if not rir.exists():
            rir = branch / f"{key}_rir.wav"

        if noisy.exists() and clean.exists():
            g = {"noisy.wav": noisy, "clean.wav": clean}
            if rir.exists():
                g["rir.wav"] = rir
            meta = branch / f"{key}.json"
            if meta.exists():
                g["json"] = meta
            return g
        return None

    if not args.card_only:
        se_summary = pack_branch_to_shards(BRANCH_SE, SHARDS_DIR / "speech_enhancement", "se", se_groups, SAMPLES_PER_SHARD)
        branch_stats["speech_enhancement"] = se_summary.get("samples_packed", 0)
        print(f">>> SE branch: {se_summary.get('samples_packed', 0)} samples across {se_summary.get('total_shards', 0)} shards")

        def clf_groups(sample_root: Path):
            key = sample_root.name
            branch = sample_root.parent
            wav = branch / "audio" / f"{key}.wav"
            if not wav.exists():
                wav = branch / f"{key}.wav"
            if wav.exists():
                g = {"wav": wav}
                meta = branch / f"{key}.json"
                if meta.exists():
                    g["json"] = meta
                return g
            return None

        clf_summary = pack_branch_to_shards(BRANCH_CLASSIFIER, SHARDS_DIR / "classifier", "clf", clf_groups, SAMPLES_PER_SHARD)
        branch_stats["classifier"] = clf_summary.get("samples_packed", 0)
        print(f">>> Classifier branch: {clf_summary.get('samples_packed', 0)} samples across {clf_summary.get('total_shards', 0)} shards")

        def aec_groups(sample_root: Path):
            key = sample_root.name
            branch = sample_root.parent
            mic = branch / "mic" / f"{key}_mic.wav"
            if not mic.exists():
                mic = branch / f"{key}_mic.wav"
            farend = branch / "farend" / f"{key}_farend.wav"
            if not farend.exists():
                farend = branch / f"{key}_farend.wav"
            nearend = branch / "nearend" / f"{key}_nearend.wav"
            if not nearend.exists():
                nearend = branch / f"{key}_nearend.wav"
            echo = branch / "echo" / f"{key}_echo.wav"
            if not echo.exists():
                echo = branch / f"{key}_echo.wav"

            if mic.exists() and farend.exists():
                g = {"mic.wav": mic, "farend.wav": farend}
                if nearend.exists():
                    g["nearend.wav"] = nearend
                if echo.exists():
                    g["echo.wav"] = echo
                meta = branch / f"{key}.json"
                if meta.exists():
                    g["json"] = meta
                return g
            return None

        aec_summary = pack_branch_to_shards(BRANCH_AEC, SHARDS_DIR / "aec", "aec", aec_groups, SAMPLES_PER_SHARD)
        branch_stats["aec"] = aec_summary.get("samples_packed", 0)
        print(f">>> AEC branch: {aec_summary.get('samples_packed', 0)} samples across {aec_summary.get('total_shards', 0)} shards")

    card_path = SHARDS_DIR / "DATASET_CARD.md"
    write_dataset_card(card_path, branch_stats)
    print(f">>> Dataset card written to {card_path}")
    print(">>> Export complete. Point PyTorch training at data/shards/ via data_forge.exporter's IterableDataset classes.")


def cmd_run_all(args):
    print("=== PROJECT AEGIS: FULL PIPELINE EXECUTION INITIATED ===")
    export_bibliography_json(MANIFESTS_DIR / "verified_bibliography.json")
    cmd_fetch(args)
    if not args.dry_run:
        cmd_preprocess(args)
        cmd_augment(args)
        cmd_mix(args)
        cmd_export(args)
    cmd_verify(args)
    print("=== PROJECT AEGIS: PIPELINE COMPLETE ===")


def main():
    parser = argparse.ArgumentParser(description="Project AEGIS Data-Forge Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # fetch
    p_fetch = subparsers.add_parser("fetch", help="Fetch dataset files")
    p_fetch.add_argument("--source", type=str, default="all", help="Source name or 'all'")
    p_fetch.add_argument("--sample-mode", action="store_true", help="Download sample subset for validation")
    p_fetch.add_argument("--full-mode", action="store_true", help="Full multi-terabyte production download on 4TB storage")
    p_fetch.add_argument("--server-mode", action="store_true", help=argparse.SUPPRESS)  # Alias for backwards compatibility
    p_fetch.add_argument("--dry-run", action="store_true", help="Verify endpoints without writing files")

    # preprocess
    p_pre = subparsers.add_parser("preprocess", help="Run 10-step preprocessing pipeline")
    p_pre.add_argument("--max-workers", type=int, default=4, help="Worker threads")
    p_pre.add_argument("--commercial-strict", action="store_true", help="Filter out CC-BY-NC files")

    # augment
    subparsers.add_parser("augment", help="Run grounded per-source augmentation")

    # mix
    p_mix = subparsers.add_parser("mix", help="Run multi-branch data-forge mixing")
    p_mix.add_argument("--num-mixtures", type=int, default=100, help="Number of mixtures to generate")
    p_mix.add_argument("--min-snr", type=float, default=-5.0, help="Minimum SNR in dB")
    p_mix.add_argument("--max-snr", type=float, default=20.0, help="Maximum SNR in dB")

    # verify
    subparsers.add_parser("verify", help="Run audit and verification report")

    # export
    p_export = subparsers.add_parser("export", help="Pack data/forge/ into WebDataset shards + dataset card (industry-standard training input)")
    p_export.add_argument("--card-only", action="store_true", help="Regenerate only the dataset card, skip re-sharding")

    # run-all
    p_all = subparsers.add_parser("run-all", help="Run complete end-to-end pipeline")
    p_all.add_argument("--source", type=str, default="all")
    p_all.add_argument("--sample-mode", action="store_true", help="Run in sample mode for testing")
    p_all.add_argument("--full-mode", action="store_true", help="Run full production pipeline on 4TB storage")
    p_all.add_argument("--server-mode", action="store_true", help=argparse.SUPPRESS)  # Alias for backwards compatibility
    p_all.add_argument("--dry-run", action="store_true", help="Dry run verification")
    p_all.add_argument("--max-workers", type=int, default=4)
    p_all.add_argument("--commercial-strict", action="store_true")
    p_all.add_argument("--num-mixtures", type=int, default=50)
    p_all.add_argument("--min-snr", type=float, default=-5.0)
    p_all.add_argument("--max-snr", type=float, default=20.0)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "fetch": cmd_fetch,
        "preprocess": cmd_preprocess,
        "augment": cmd_augment,
        "mix": cmd_mix,
        "verify": cmd_verify,
        "export": cmd_export,
        "run-all": cmd_run_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
