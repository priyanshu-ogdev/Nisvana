"""training/scripts/train_se_escalation.py — entry point for Model 2 (aegis-se-escalation)"""
import argparse
from training.configs.se_escalation_config import SeEscalationConfig
from training.data.weighted_shard_sampler import build_weighted_se_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = SeEscalationConfig()
    if args.resume:
        config.resume_from = args.resume

    try:
        train_ds = build_weighted_se_dataset(
            config.data.speech_enhancement_shards, "train", config.class_oversample_factors
        )
    except ImportError:
        train_ds = None
    print(f"[{config.model_key}] lookahead=stock(2/2) lr={config.lr} "
          f"snr_weights={dict(zip(config.dataloader_snrs, config.dataloader_snr_weights))}")

    from training.trainers.se_escalation_trainer import SeEscalationTrainer
    trainer = SeEscalationTrainer(config=config, train_dataset=train_ds)
    print(f"[{config.model_key}] Initialized SeEscalationTrainer (step={trainer.step}). Ready for training.")
    return trainer


if __name__ == "__main__":
    main()
