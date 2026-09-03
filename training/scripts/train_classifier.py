"""training/scripts/train_classifier.py — entry point for Model 4 (aegis-clf-gate)"""
import argparse
from training.configs.classifier_config import ClassifierConfig, GATE_CLASSES
from data_forge.exporter import AegisClassifierIterableDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = ClassifierConfig()
    if args.resume:
        config.resume_from = args.resume

    try:
        train_ds = AegisClassifierIterableDataset(config.data.classifier_shards, split="train")
    except ImportError:
        train_ds = None
    print(f"[{config.model_key}] gate_classes={GATE_CLASSES} arch={config.architecture} "
          f"lr={config.lr} class_loss_weights={config.class_loss_weights}")
    print(f"[{config.model_key}] validate_against_se_gap={config.validate_against_se_gap} "
          f"(integration test against Models 1-2's measured gap, not just standalone accuracy)")

    from training.trainers.classifier_trainer import ClassifierTrainer
    trainer = ClassifierTrainer(config=config, train_dataset=train_ds)
    print(f"[{config.model_key}] Initialized ClassifierTrainer (step={trainer.step}). Ready for training.")
    return trainer


if __name__ == "__main__":
    main()
