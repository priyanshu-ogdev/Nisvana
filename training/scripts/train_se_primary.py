"""
training/scripts/train_se_primary.py — entry point for Model 1 (aegis-se-primary)

Usage: python -m training.scripts.train_se_primary [--resume PATH]
"""
import argparse
from training.configs.se_primary_config import SePrimaryConfig
from training.data.weighted_shard_sampler import build_weighted_se_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = SePrimaryConfig()
    if args.resume:
        config.resume_from = args.resume

    try:
        train_ds = build_weighted_se_dataset(
            config.data.speech_enhancement_shards, "train", config.class_oversample_factors
        )
        val_ds = build_weighted_se_dataset(
            config.data.speech_enhancement_shards, "val", config.class_oversample_factors
        )
    except ImportError:
        train_ds, val_ds = None, None

    print(f"[{config.model_key}] df_lookahead={config.df_lookahead} conv_lookahead={config.conv_lookahead} "
          f"lr={config.lr} max_epochs={config.max_epochs}")
    print(f"[{config.model_key}] class_oversample_factors={config.class_oversample_factors}")

    from training.trainers.se_primary_trainer import SePrimaryTrainer
    trainer = SePrimaryTrainer(config=config, train_dataset=train_ds, val_dataset=val_ds)
    print(f"[{config.model_key}] Initialized SePrimaryTrainer (step={trainer.step}). Ready for training.")
    return trainer


if __name__ == "__main__":
    main()
