"""training/scripts/train_se_crosscheck.py — entry point for Model 3 (aegis-se-crosscheck / CleanUMamba)"""
import argparse
from training.configs.se_crosscheck_config import SeCrosscheckConfig
from training.data.weighted_shard_sampler import build_weighted_se_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = SeCrosscheckConfig()
    if args.resume:
        config.resume_from = args.resume

    try:
        train_ds = build_weighted_se_dataset(
            config.data.speech_enhancement_shards, "train", config.class_oversample_factors
        )
    except ImportError:
        train_ds = None
    print(f"[{config.model_key}] target_param_count={config.target_param_count} "
          f"total_finetune_steps={config.total_finetune_steps} batch_size={config.batch_size}")
    print(f"[{config.model_key}] REMINDER: evaluate at 48kHz on OUR shards -- "
          f"do not cite the paper's 16kHz DNS-2020 numbers as this model's baseline.")

    from training.trainers.se_crosscheck_trainer import SeCrosscheckTrainer
    trainer = SeCrosscheckTrainer(config=config, train_dataset=train_ds)
    print(f"[{config.model_key}] Initialized SeCrosscheckTrainer (step={trainer.step}). Ready for training.")
    return trainer


if __name__ == "__main__":
    main()
