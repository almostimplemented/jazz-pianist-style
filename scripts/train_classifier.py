#!/usr/bin/env python3
"""Fine-tune Aria as a pianist classifier (Section 4.1 of the paper).

Full fine-tuning of the Aria medium backbone with a linear head over the
end-of-sequence hidden state, on 1024-token chunks. Defaults are the paper's
recipe: learning rate 1e-5, batch size 16, 15 epochs with early stopping on
validation loss (patience 5), class-balanced cross-entropy, and pitch / tempo /
velocity augmentation with probability 0.5. The same recipe trains the
PiJAMA-12, PiJAMA-30 and synthetic-only classifiers; only the data changes.

Base weights come from the Hugging Face Hub (loubb/aria-medium-base) unless
--base-checkpoint is given.

Example:
    python scripts/train_classifier.py \
        --train-jsonl data/pijama12_1024/train.jsonl \
        --val-jsonl data/pijama12_1024/val.jsonl \
        --artist-map data/pijama12_1024/artist_to_id.json \
        --out-dir checkpoints/classifier_run
    # -> checkpoints/classifier_run/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from llama_pijama.training.pijama_classifier import train


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--artist-map", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--base-checkpoint", type=Path, default=None,
                    help="Aria weights to start from (default: download aria-medium-base)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-acc-steps", type=int, default=1)
    ap.add_argument("--augmentation-prob", type=float, default=0.5)
    ap.add_argument("--no-augmentation", action="store_true")
    ap.add_argument("--no-balanced-loss", action="store_true")
    ap.add_argument("--early-stopping-patience", type=int, default=5)
    ap.add_argument("--freeze-base", action="store_true",
                    help="train only the head (a linear probe; not the paper's recipe)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    return ap.parse_args()


def main():
    args = parse_args()
    train(
        model_name=args.model_name,
        train_data_path=str(args.train_jsonl),
        val_data_path=str(args.val_jsonl),
        artist_to_id_path=str(args.artist_map),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grad_acc_steps=args.grad_acc_steps,
        project_dir=str(args.out_dir),
        checkpoint_path=str(args.base_checkpoint) if args.base_checkpoint else None,
        freeze_base=args.freeze_base,
        use_pretrained=True,
        use_wandb=args.wandb,
        balanced_loss=not args.no_balanced_loss,
        use_augmentation=not args.no_augmentation,
        augmentation_prob=args.augmentation_prob,
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    main()
