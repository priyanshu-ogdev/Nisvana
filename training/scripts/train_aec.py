"""
training/scripts/train_aec.py — entry point for Model 5 (aegis-aec-gate)

Refuses to run without --force, since train_by_default=False in the
config -- the base implementation plan uses the existing deepvqe-ggml
checkpoint as-is. This script exists for naming/structural consistency
and as the defined integration point if AEC fine-tuning is explicitly
requested later.
"""
import argparse
import sys
from training.configs.aec_config import AecGateConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Required: this model is not trained by default (see config.train_by_default).")
    args = parser.parse_args()

    config = AecGateConfig()
    if not config.train_by_default and not args.force:
        print(f"[{config.model_key}] train_by_default=False. Using existing checkpoint "
              f"'{config.pretrained_checkpoint}' as-is, per the base implementation plan. "
              f"Pass --force to override (hyperparameters below are PLACEHOLDERS, not "
              f"literature-grounded -- see aec_config.py's docstring).")
        sys.exit(0)

    print(f"[{config.model_key}] FORCED fine-tune run with placeholder hyperparameters: "
          f"lr={config.lr_placeholder} batch_size={config.batch_size_placeholder} "
          f"data_source={config.data_source}")


if __name__ == "__main__":
    main()
