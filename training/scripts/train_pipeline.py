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
from training.data.dataset_guard import (
    make_loader_kwargs,
    seed_everything,
    validate_split_shards,
)


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
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override total_finetune_steps by epochs (converted via steps_per_epoch, default 1000/epoch). "
             "Every AEGIS model config is step-based (total_finetune_steps), not epoch-based -- this flag "
             "is a convenience conversion, not a native config field."
    )
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
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="DataLoader workers. Defaults to the model config; workers are deterministically seeded.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Global reproducibility seed.")
    return parser.parse_args()


def _apply_epoch_override(config, args) -> None:
    """
    Converts --epochs into config.total_finetune_steps, the field every
    AEGIS BaseModelConfig subclass actually declares. There is no
    `max_epochs` field anywhere in the config schema -- assigning one
    directly (the old behavior) either silently did nothing useful or,
    when read back unconditionally without --epochs being passed,
    crashed with AttributeError before the training loop ever started.
    """
    if args.epochs:
        steps_per_epoch = getattr(config, "steps_per_epoch", 1000)
        config.total_finetune_steps = args.epochs * steps_per_epoch
    if args.num_workers is not None:
        config.num_workers = max(0, args.num_workers)
    config.gradient_accumulation_steps = max(1, args.grad_accum)
    config.seed = args.seed


def _validate_se_data(config, args) -> None:
    """Fail before model construction when a required real split is absent."""
    if args.dry_run or not config.data.require_real_shards:
        return
    validate_split_shards(
        [config.data.speech_enhancement_shards],
        required_splits=config.data.required_splits,
    )


def se_collate_fn(batch):
    if not batch:
        return {}
    import io, numpy as np, soundfile as sf
    cleaned = []
    for sample in batch:
        s = {}
        for k in ("noisy.wav", "clean.wav", "rir.wav", "noisy", "clean"):
            v = sample.get(k)
            if v is None:
                continue
            if isinstance(v, (bytes, bytearray)):
                try:
                    v, _ = sf.read(io.BytesIO(v), dtype="float32")
                except Exception:
                    continue
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v.copy())
            elif not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            v = v.squeeze()
            if v.ndim > 1:
                if v.shape[0] <= 8 and v.shape[-1] > v.shape[0]:
                    v = v.mean(dim=0)
                else:
                    v = v.mean(dim=-1)
            v = torch.atleast_1d(v)
            s[k] = v
        cleaned.append((s, sample.get("json", {})))

    max_len = 2048
    for s, _ in cleaned:
        for k in ("noisy.wav", "clean.wav", "noisy", "clean"):
            if k in s and isinstance(s[k], torch.Tensor):
                max_len = max(max_len, s[k].shape[-1])
    max_len = min(max_len, 192000)

    batch_noisy, batch_clean, batch_json = [], [], []
    for s, meta in cleaned:
        noisy = s.get("noisy.wav", s.get("noisy"))
        clean = s.get("clean.wav", s.get("clean"))
        if noisy is None or clean is None:
            continue
        if noisy.shape[-1] > max_len:
            noisy = noisy[:max_len]
        elif noisy.shape[-1] < max_len:
            noisy = torch.nn.functional.pad(noisy, (0, max_len - noisy.shape[-1]))
        if clean.shape[-1] > max_len:
            clean = clean[:max_len]
        elif clean.shape[-1] < max_len:
            clean = torch.nn.functional.pad(clean, (0, max_len - clean.shape[-1]))
        batch_noisy.append(noisy)
        batch_clean.append(clean)
        batch_json.append(meta)

    if not batch_noisy:
        return {}
    stacked_noisy = torch.stack(batch_noisy, dim=0)
    stacked_clean = torch.stack(batch_clean, dim=0)
    return {
        "noisy.wav": stacked_noisy,
        "clean.wav": stacked_clean,
        "noisy": stacked_noisy,
        "clean": stacked_clean,
        "json": batch_json,
    }


def clf_collate_fn(batch):
    if not batch:
        return {}
    import io, json, numpy as np, soundfile as sf
    target_len = 9600
    batch_wav, batch_label, batch_json = [], [], []

    for sample in batch:
        audio = sample.get("wav")
        if audio is None:
            audio = sample.get("wav.wav")
        if audio is None:
            audio = sample.get("audio")
        if audio is None:
            continue
        if isinstance(audio, (bytes, bytearray)):
            try:
                audio, _ = sf.read(io.BytesIO(audio), dtype="float32")
            except Exception:
                continue
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.copy())
        elif not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        audio = audio.squeeze()
        if audio.ndim > 1:
            if audio.shape[0] <= 8 and audio.shape[-1] > audio.shape[0]:
                audio = audio.mean(dim=0)
            else:
                audio = audio.mean(dim=-1)
        audio = torch.atleast_1d(audio)

        if audio.shape[-1] > target_len:
            audio = audio[:target_len]
        elif audio.shape[-1] < target_len:
            audio = torch.nn.functional.pad(audio, (0, target_len - audio.shape[-1]))

        peak = torch.max(torch.abs(audio))
        if peak > 1.0:
            audio = audio / peak

        cat_idx = sample.get("category_index", sample.get("label"))
        meta = sample.get("json", {})
        if isinstance(meta, (bytes, bytearray)):
            try:
                meta = json.loads(meta.decode("utf-8"))
            except Exception:
                meta = {}
        elif isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        if cat_idx is None and isinstance(meta, dict):
            cat_idx = meta.get("category_index")
            if cat_idx is None:
                u_class = meta.get("unified_class", "general_noise")
                from training.configs.classifier_config import UNIFIED_TO_GATE_CLASS
                gate_name = UNIFIED_TO_GATE_CLASS.get(u_class, "harmonic")
                gate_map = {"harmonic": 0, "impulsive": 1, "speech_dominant": 2}
                cat_idx = gate_map.get(gate_name, 0)
        if cat_idx is None:
            cat_idx = 0

        batch_wav.append(audio)
        batch_label.append(int(cat_idx))
        batch_json.append(meta)

    if not batch_wav:
        return {}

    return {
        "wav": torch.stack(batch_wav, dim=0),
        "label": torch.tensor(batch_label, dtype=torch.long),
        "category_index": torch.tensor(batch_label, dtype=torch.long),
        "json": batch_json,
    }


def aec_collate_fn(batch):
    if not batch:
        return {}
    import io, numpy as np, soundfile as sf
    cleaned = []
    for sample in batch:
        s = {}
        for k in ("mic.wav", "farend.wav", "nearend.wav", "echo.wav", "mic", "farend", "nearend"):
            v = sample.get(k)
            if v is None:
                continue
            if isinstance(v, (bytes, bytearray)):
                try:
                    v, _ = sf.read(io.BytesIO(v), dtype="float32")
                except Exception:
                    continue
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v.copy())
            elif not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            v = v.squeeze()
            if v.ndim > 1:
                if v.shape[0] <= 8 and v.shape[-1] > v.shape[0]:
                    v = v.mean(dim=0)
                else:
                    v = v.mean(dim=-1)
            v = torch.atleast_1d(v)
            s[k] = v
        cleaned.append(s)

    max_len = 2048
    for s in cleaned:
        for k in ("mic.wav", "farend.wav", "nearend.wav", "mic", "farend", "nearend"):
            if k in s and isinstance(s[k], torch.Tensor):
                max_len = max(max_len, s[k].shape[-1])
    max_len = min(max_len, 192000)

    batch_mic, batch_farend, batch_nearend = [], [], []
    for s in cleaned:
        mic = s.get("mic.wav", s.get("mic"))
        farend = s.get("farend.wav", s.get("farend"))
        nearend = s.get("nearend.wav", s.get("nearend", mic))
        if mic is None or farend is None:
            continue
        if mic.shape[-1] > max_len:
            mic = mic[:max_len]
        elif mic.shape[-1] < max_len:
            mic = torch.nn.functional.pad(mic, (0, max_len - mic.shape[-1]))
        if farend.shape[-1] > max_len:
            farend = farend[:max_len]
        elif farend.shape[-1] < max_len:
            farend = torch.nn.functional.pad(farend, (0, max_len - farend.shape[-1]))
        if nearend.shape[-1] > max_len:
            nearend = nearend[:max_len]
        elif nearend.shape[-1] < max_len:
            nearend = torch.nn.functional.pad(nearend, (0, max_len - nearend.shape[-1]))
        batch_mic.append(mic)
        batch_farend.append(farend)
        batch_nearend.append(nearend)

    if not batch_mic:
        return {}
    return {
        "mic.wav": torch.stack(batch_mic, dim=0),
        "farend.wav": torch.stack(batch_farend, dim=0),
        "nearend.wav": torch.stack(batch_nearend, dim=0),
    }


def train_se_crosscheck(args) -> Path:
    """Model 3: CleanUMamba SSM Crosscheck (Teacher Model)."""
    print("\n" + "=" * 80)
    print(">>> [STAGE 1/5] TRAINING MODEL 3: CLEANUMAMBA SSM (TEACHER)")
    print("=" * 80)
    config = SeCrosscheckConfig()
    _apply_epoch_override(config, args)
    if args.lr:
        config.lr = args.lr
    if args.resume:
        config.resume_from = args.resume
    config.precision = args.precision

    _validate_se_data(config, args)
    if args.dry_run:
        train_ds = None
    else:
        try:
            train_ds = build_weighted_se_dataset(
                config.data.speech_enhancement_shards, "train", config.class_oversample_factors
            )
        except (ImportError, FileNotFoundError):
            train_ds = None

    from training.trainers.se_crosscheck_trainer import SeCrosscheckTrainer
    trainer = SeCrosscheckTrainer(config=config, train_dataset=train_ds)
    print(f"[{config.model_key}] Initialized CleanUMamba teacher (precision={config.precision}, device={args.device}).")

    if not args.dry_run:
        if train_ds:
            print(f"[{config.model_key}] Starting training loop for {config.total_finetune_steps} steps...")
            from torch.utils.data import DataLoader
            batch_size = args.batch_size or getattr(config, "batch_size", 16)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, collate_fn=se_collate_fn,
                **make_loader_kwargs(config),
            )
            loop_result = trainer.run_training_loop(train_loader, epochs=args.epochs)
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
    _apply_epoch_override(config, args)
    if args.lr:
        config.lr = args.lr
    if args.resume:
        config.resume_from = args.resume
    config.precision = args.precision
    config.qat_enabled = args.qat
    config.distillation_factor = args.distillation_factor if args.distillation else 0.0

    _validate_se_data(config, args)
    if args.dry_run:
        train_ds, val_ds = None, None
    else:
        try:
            train_ds = build_weighted_se_dataset(
                config.data.speech_enhancement_shards, "train", config.class_oversample_factors
            )
            val_ds = build_weighted_se_dataset(
                config.data.speech_enhancement_shards, "val", config.class_oversample_factors
            )
        except (ImportError, FileNotFoundError):
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
            print(f"[{config.model_key}] Starting training loop for {config.total_finetune_steps} steps...")
            from torch.utils.data import DataLoader
            batch_size = args.batch_size or getattr(config, "batch_size", 16)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, collate_fn=se_collate_fn,
                **make_loader_kwargs(config),
            )
            val_loader = (
                DataLoader(
                    val_ds, batch_size=batch_size, collate_fn=se_collate_fn,
                    **make_loader_kwargs(config),
                )
                if val_ds else None
            )
            loop_result = trainer.run_training_loop(train_loader, val_loader, epochs=args.epochs)
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
    _apply_epoch_override(config, args)
    if args.lr:
        config.lr = args.lr
    if args.resume:
        config.resume_from = args.resume
    config.precision = args.precision
    config.qat_enabled = args.qat

    _validate_se_data(config, args)
    try:
        if args.dry_run:
            raise FileNotFoundError("dry-run skips dataset construction")
        train_ds = build_weighted_se_dataset(
            config.data.speech_enhancement_shards, "train", config.class_oversample_factors
        )
    except (ImportError, FileNotFoundError):
        train_ds = None

    from training.trainers.se_escalation_trainer import SeEscalationTrainer
    trainer = SeEscalationTrainer(config=config, train_dataset=train_ds)
    print(f"[{config.model_key}] Initialized SeEscalationTrainer with 1-chunk lookahead delay buffer.")

    if not args.dry_run:
        if train_ds:
            print(f"[{config.model_key}] Starting training loop for {config.total_finetune_steps} steps...")
            from torch.utils.data import DataLoader
            batch_size = args.batch_size or getattr(config, "batch_size", 16)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, collate_fn=se_collate_fn,
                **make_loader_kwargs(config),
            )
            loop_result = trainer.run_training_loop(train_loader, epochs=args.epochs)
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
    _apply_epoch_override(config, args)
    if args.lr:
        config.lr = args.lr
    if args.resume:
        config.resume_from = args.resume

    try:
        if args.dry_run:
            raise FileNotFoundError("dry-run skips dataset construction")
        from data_forge.exporter import AegisClassifierIterableDataset
        train_ds = AegisClassifierIterableDataset(config.data.classifier_shards, split="train")
    except (ImportError, FileNotFoundError):
        train_ds = None

    from training.trainers.classifier_trainer import ClassifierTrainer
    trainer = ClassifierTrainer(config=config, train_dataset=train_ds)
    print(f"[{config.model_key}] Initialized ClassifierTrainer on 0.2s windows (arch={config.architecture}).")

    if not args.dry_run:
        if train_ds:
            print(f"[{config.model_key}] Starting training loop for {config.total_finetune_steps} steps...")
            from torch.utils.data import DataLoader
            batch_size = args.batch_size or getattr(config, "batch_size", 32)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, collate_fn=clf_collate_fn,
                **make_loader_kwargs(config),
            )
            loop_result = trainer.run_training_loop(train_loader, epochs=args.epochs)
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
    _apply_epoch_override(config, args)
    try:
        if args.dry_run:
            raise FileNotFoundError("dry-run skips dataset construction")
        from data_forge.exporter import AegisAecIterableDataset
        train_ds = AegisAecIterableDataset(config.data.aec_shards, split="train")
    except (ImportError, FileNotFoundError):
        train_ds = None

    from training.trainers.aec_trainer import AecGateTrainer
    trainer = AecGateTrainer(config=config, train_dataset=train_ds)

    if not args.dry_run:
        if train_ds:
            print(f"[{config.model_key}] Starting training loop for {config.total_finetune_steps} steps...")
            from torch.utils.data import DataLoader
            batch_size = args.batch_size or getattr(config, "batch_size", 16)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, collate_fn=aec_collate_fn,
                **make_loader_kwargs(config),
            )
            loop_result = trainer.run_training_loop(train_loader, epochs=args.epochs)
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
    seed_everything(args.seed)
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