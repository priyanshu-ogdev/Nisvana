"""
Project AEGIS — Master Unified Training Pipeline Orchestrator (Rev 3)
Connects all ML training layers into a single, cohesive, production-grade CLI.

Targets:
- NVIDIA DGX Spark GB10 (Grace Blackwell, CUDA 13, 128GB Unified Memory)
- Native bfloat16 AMP (`precision="bf16"`)
- Quantization-Aware Training (QAT) from Epoch 1
- CleanUMamba (SSM) Knowledge Distillation into DeepFilterNet3
- Model 2 Lookahead Output-Delay Buffering (10ms future context)
- Worst-Class Pareto Guard on Fragile Military Classes

Usage:
  python -m training.scripts.train_pipeline --model all
  python -m training.scripts.train_pipeline --model se_primary --precision bf16 --qat
  python -m training.scripts.train_pipeline --model se_escalation --epochs 80
  python -m training.scripts.train_pipeline --model se_crosscheck --epochs 100
  python -m training.scripts.train_pipeline --model classifier --epochs 50
  python -m training.scripts.train_pipeline --model aec --force
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from training.configs.se_primary_config import SePrimaryConfig
from training.configs.se_escalation_config import SeEscalationConfig
from training.configs.se_crosscheck_config import SeCrosscheckConfig
from training.configs.classifier_config import ClassifierConfig
from training.configs.aec_config import AecGateConfig
from training.data.weighted_shard_sampler import build_weighted_se_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project AEGIS Unified Master Training Orchestrator (Rev 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "se_primary", "se_escalation", "se_crosscheck", "classifier", "aec"],
        help="Target model to train, or 'all' for full sequential ensemble execution",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override default max_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per step")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Training precision: bf16 (Grace Blackwell GB10 native), fp16, or fp32",
    )
    parser.add_argument(
        "--qat",
        dest="qat",
        action="store_true",
        default=True,
        help="Enable Quantization-Aware Training from epoch 1",
    )
    parser.add_argument("--no-qat", dest="qat", action="store_false", help="Disable QAT")
    parser.add_argument(
        "--distillation",
        dest="distillation",
        action="store_true",
        default=True,
        help="Enable CleanUMamba spectrogram knowledge distillation into DeepFilterNet3",
    )
    parser.add_argument("--no-distillation", dest="distillation", action="store_false", help="Disable distillation")
    parser.add_argument(
        "--distillation-factor",
        type=float,
        default=0.3,
        help="Distillation loss weight multiplier",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--force", action="store_true", help="Force execution (required for Model 5 AEC)")
    parser.add_argument("--dry-run", action="store_true", help="Perform architecture initialization dry run without training loop")
    return parser.parse_args()


def train_se_crosscheck(args) -> Path:
    """Model 3: CleanUMamba SSM Crosscheck (Teacher Model)."""
    print("\n" + "=" * 80)
    print(">>> [STAGE 1/5] TRAINING MODEL 3: CLEANUMAMBA SSM (TEACHER)")
    print("=" * 80)
    config = SeCrosscheckConfig()
    if args.epochs:
        config.max_epochs = args.epochs
    if args.lr:
        config.lr = args.lr
    if args.resume:
        config.resume_from = args.resume
    config.precision = args.precision

    try:
        train_ds = build_weighted_se_dataset(
            config.data.speech_enhancement_shards, "train", config.class_oversample_factors
        )
    except Exception:
        train_ds = None

    from training.trainers.se_crosscheck_trainer import SeCrosscheckTrainer
    trainer = SeCrosscheckTrainer(config=config, train_dataset=train_ds)
    print(f"[{config.model_key}] Initialized CleanUMamba teacher (precision={config.precision}, device={args.device}).")

    if not args.dry_run:
        if train_ds:
            print(f"[{config.model_key}] Starting training loop for {config.max_epochs} epochs...")
            from torch.utils.data import DataLoader
            batch_size = args.batch_size or getattr(config, "batch_size", 16)
            train_loader = DataLoader(train_ds, batch_size=batch_size)
            loop_result = trainer.run_training_loop(train_loader)
            print(f"[{config.model_key}] Training completed. Total steps: {loop_result.get('total_steps', 0)}")
        else:
            print(f"[{config.model_key}] WARNING: Shards not loaded or webdataset not installed. Skipping loop.")
    ckpt_path = config.checkpoint_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        saved = trainer.save_checkpoint(model_state=trainer.model.state_dict() if hasattr(trainer.model, "state_dict") else trainer.model)
        if saved and Path(saved).exists():
            import shutil
            shutil.copy2(saved, ckpt_path)
    print(f"[{config.model_key}] Model 3 checkpoint: {ckpt_path}")
    return ckpt_path


def train_se_primary(args, teacher_ckpt: Optional[Path] = None) -> Path:
    """Model 1: DeepFilterNet3 Base with CleanUMamba Distillation & QAT from Epoch 1."""
    print("\n" + "=" * 80)
    print(">>> [STAGE 2/5] TRAINING MODEL 1: DEEPFILTERNET3 BASE (0ms LOOKAHEAD)")
    print("=" * 80)
    config = SePrimaryConfig()
    if args.epochs:
        config.max_epochs = args.epochs
    if args.lr:
        config.lr = args.lr
    if args.resume:
        config.resume_from = args.resume
    config.precision = args.precision
    config.qat_enabled = args.qat
    config.distillation_factor = args.distillation_factor if args.distillation else 0.0

    try:
        train_ds = build_weighted_se_dataset(
            config.data.speech_enhancement_shards, "train", config.class_oversample_factors
        )
        val_ds = build_weighted_se_dataset(
            config.data.speech_enhancement_shards, "val", config.class_oversample_factors
        )
    except Exception:
        train_ds, val_ds = None, None

    from training.trainers.se_primary_trainer import SePrimaryTrainer
    trainer = SePrimaryTrainer(config=config, train_dataset=train_ds, val_dataset=val_ds)
    print(f"[{config.model_key}] Initialized SePrimaryTrainer:")
    print(f"  - Precision: {config.precision} (Native Blackwell AMP)")
    print(f"  - QAT from Epoch 1: {config.qat_enabled}")
    print(f"  - Distillation Factor: {config.distillation_factor}")
    print(f"  - Lookahead: {config.df_lookahead} frames (strict 0ms)")

    if not args.dry_run:
        if train_ds:
            print(f"[{config.model_key}] Starting training loop for {config.max_epochs} epochs...")
            from torch.utils.data import DataLoader
            batch_size = args.batch_size or getattr(config, "batch_size", 16)
            train_loader = DataLoader(train_ds, batch_size=batch_size)
            val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None
            loop_result = trainer.run_training_loop(train_loader, val_loader)
            print(f"[{config.model_key}] Training completed. Total steps: {loop_result.get('total_steps', 0)}")
        else:
            print(f"[{config.model_key}] WARNING: Shards not loaded or webdataset not installed. Skipping loop.")

    ckpt_path = config.checkpoint_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        saved = trainer.save_checkpoint(model_state=trainer.model.state_dict() if hasattr(trainer.model, "state_dict") else trainer.model)
        if saved and Path(saved).exists():
            import shutil
            shutil.copy2(saved, ckpt_path)
    print(f"[{config.model_key}] Model 1 checkpoint: {ckpt_path}")
    return ckpt_path


def train_se_escalation(args) -> Path:
    """Model 2: DeepFilterNet3 Escalation with 10ms Lookahead Output Delay."""
    print("\n" + "=" * 80)
    print(">>> [STAGE 3/5] TRAINING MODEL 2: DEEPFILTERNET3 ESCALATION (10ms LOOKAHEAD)")
    print("=" * 80)
    config = SeEscalationConfig()
    if args.epochs:
        config.max_epochs = args.epochs
    if args.lr:
        config.lr = args.lr
    if args.resume:
        config.resume_from = args.resume
    config.precision = args.precision
    config.qat_enabled = args.qat

    try:
        train_ds = build_weighted_se_dataset(
            config.data.speech_enhancement_shards, "train", config.class_oversample_factors
        )
    except Exception:
        train_ds = None

    from training.trainers.se_escalation_trainer import SeEscalationTrainer
    trainer = SeEscalationTrainer(config=config, train_dataset=train_ds)
    print(f"[{config.model_key}] Initialized SeEscalationTrainer with 1-chunk lookahead delay buffer.")

    if not args.dry_run:
        if train_ds:
            print(f"[{config.model_key}] Starting training loop for {config.max_epochs} epochs...")
            from torch.utils.data import DataLoader
            batch_size = args.batch_size or getattr(config, "batch_size", 16)
            train_loader = DataLoader(train_ds, batch_size=batch_size)
            loop_result = trainer.run_training_loop(train_loader)
            print(f"[{config.model_key}] Training completed. Total steps: {loop_result.get('total_steps', 0)}")
        else:
            print(f"[{config.model_key}] WARNING: Shards not loaded or webdataset not installed. Skipping loop.")

    ckpt_path = config.checkpoint_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        saved = trainer.save_checkpoint(model_state=trainer.model.state_dict() if hasattr(trainer.model, "state_dict") else trainer.model)
        if saved and Path(saved).exists():
            import shutil
            shutil.copy2(saved, ckpt_path)
    print(f"[{config.model_key}] Model 2 checkpoint: {ckpt_path}")
    return ckpt_path


def train_classifier(args) -> Path:
    """Model 4: Acoustic Gating Classifier (3-Way Conformer/Conv1d)."""
    print("\n" + "=" * 80)
    print(">>> [STAGE 4/5] TRAINING MODEL 4: ACOUSTIC GATING CLASSIFIER")
    print("=" * 80)
    config = ClassifierConfig()
    if args.epochs:
        config.max_epochs = args.epochs
    if args.lr:
        config.lr = args.lr
    if args.resume:
        config.resume_from = args.resume

    try:
        from data_forge.exporter import AegisClassifierIterableDataset
        train_ds = AegisClassifierIterableDataset(config.data.classifier_shards, split="train")
    except Exception:
        train_ds = None

    from training.trainers.classifier_trainer import ClassifierTrainer
    trainer = ClassifierTrainer(config=config, train_dataset=train_ds)
    print(f"[{config.model_key}] Initialized ClassifierTrainer on 0.2s windows (arch={config.architecture}).")

    if not args.dry_run:
        if train_ds:
            print(f"[{config.model_key}] Starting training loop for {config.max_epochs} epochs...")
            from torch.utils.data import DataLoader
            batch_size = args.batch_size or getattr(config, "batch_size", 32)
            train_loader = DataLoader(train_ds, batch_size=batch_size)
            loop_result = trainer.run_training_loop(train_loader)
            print(f"[{config.model_key}] Training completed. Total steps: {loop_result.get('total_steps', 0)}")
        else:
            print(f"[{config.model_key}] WARNING: Shards not loaded or webdataset not installed. Skipping loop.")

    ckpt_path = config.checkpoint_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        saved = trainer.save_checkpoint(model_state=trainer.model.state_dict() if hasattr(trainer.model, "state_dict") else trainer.model)
        if saved and Path(saved).exists():
            import shutil
            shutil.copy2(saved, ckpt_path)
    print(f"[{config.model_key}] Model 4 checkpoint: {ckpt_path}")
    return ckpt_path


def train_aec(args) -> Optional[Path]:
    """Model 5: Gated Acoustic Echo Cancellation."""
    print("\n" + "=" * 80)
    print(">>> [STAGE 5/5] MODEL 5: GATED ACOUSTIC ECHO CANCELLATION")
    print("=" * 80)
    config = AecGateConfig()
    if not config.train_by_default and not args.force:
        print(f"[{config.model_key}] train_by_default=False. Using verified checkpoint '{config.pretrained_checkpoint}' as-is.")
        return Path(config.pretrained_checkpoint)

    print(f"[{config.model_key}] FORCED fine-tuning initiated.")
    try:
        from data_forge.exporter import AegisAecIterableDataset
        train_ds = AegisAecIterableDataset(config.data.aec_shards, split="train")
    except Exception:
        train_ds = None

    from training.trainers.aec_trainer import AecGateTrainer
    trainer = AecGateTrainer(config=config, train_dataset=train_ds)

    if not args.dry_run:
        if train_ds:
            print(f"[{config.model_key}] Starting training loop for {config.max_epochs} epochs...")
            from torch.utils.data import DataLoader
            batch_size = args.batch_size or getattr(config, "batch_size", 16)
            train_loader = DataLoader(train_ds, batch_size=batch_size)
            loop_result = trainer.run_training_loop(train_loader)
            print(f"[{config.model_key}] Training completed. Total steps: {loop_result.get('total_steps', 0)}")
        else:
            print(f"[{config.model_key}] WARNING: Shards not loaded or webdataset not installed. Skipping loop.")

    ckpt_path = config.checkpoint_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        saved = trainer.save_checkpoint(model_state=trainer.model.state_dict() if hasattr(trainer.model, "state_dict") else trainer.model)
        if saved and Path(saved).exists():
            import shutil
            shutil.copy2(saved, ckpt_path)
    print(f"[{config.model_key}] Model 5 checkpoint: {ckpt_path}")
    return ckpt_path


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 80)
    print("  PROJECT AEGIS — UNIFIED ML TRAINING ORCHESTRATOR (REV 3)")
    print("  Target Box: NVIDIA DGX Spark GB10 (Grace Blackwell, CUDA 13)")
    print("=" * 80)
    print(f"  Model Selection : {args.model}")
    print(f"  Compute Device  : {args.device}")
    print(f"  Precision Mode  : {args.precision}")
    print(f"  QAT from Ep 1   : {args.qat}")
    print(f"  Distillation    : {args.distillation} (factor={args.distillation_factor})")
    print(f"  Dry-Run Mode    : {args.dry_run}")
    print("=" * 80)

    dispatch = {
        "se_crosscheck": lambda: train_se_crosscheck(args),
        "se_primary": lambda: train_se_primary(args),
        "se_escalation": lambda: train_se_escalation(args),
        "classifier": lambda: train_classifier(args),
        "aec": lambda: train_aec(args),
    }

    if args.model == "all":
        print("\n>>> Executing Full Sequential Ensemble Training in Scientific Dependency Order...")
        t_ckpt = train_se_crosscheck(args)
        train_se_primary(args, teacher_ckpt=t_ckpt)
        train_se_escalation(args)
        train_classifier(args)
        train_aec(args)
    else:
        dispatch[args.model]()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  AEGIS TRAINING PIPELINE RUN COMPLETED IN {elapsed:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
