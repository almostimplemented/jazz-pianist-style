#!/usr/bin/env python3
"""Stage the three Hugging Face model repos: weights, a clean config, a card.

Copies only weight files, writes config.json from the hyperparameters below,
and adds the matching model card from cards/. Training-time config files are
never copied: they record the run's environment, not just the model.

Example:
    python release/huggingface/stage_bundles.py \\
        --generator checkpoints/generator_best \\
        --classifier checkpoints/pijama12_classifier.pt \\
        --synthetic-classifier checkpoints/synth_classifier/best.pt
    release/huggingface/upload.sh
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent

GENERATOR_CONFIG = {
    "architecture": "CrossAttentionTransformerLM", "base_model": "aria-medium",
    "cross_attention_layers": [8, 9, 10, 11, 12, 13, 14, 15], "context_length": 4,
    "cross_attention_dropout": 0.1, "gate_init": 0.1, "embedding_dropout": 0.3,
    "num_artists": 12, "vocab_size": 17727, "max_seq_len": 4096,
    "training": {"epochs": 15, "batch_size": 4, "gradient_accumulation_steps": 8,
                 "learning_rate": 5e-6, "weight_decay": 0.02, "warmup_epochs": 0.5,
                 "mixed_precision": "fp16"},
}
CLASSIFIER_CONFIG = {"architecture": "TransformerCL (Aria medium backbone)",
                     "num_classes": 12, "max_seq_len": 1024}

# Anything that looks like a credential fails the staging outright
SECRET = re.compile(r"ghp_|github_pat_|wandb_v1_|hf_[A-Za-z0-9]{20,}|secret_key|access_key|"
                    r"aws_|endpoint_url|password", re.I)


def stage(name, files, config, card, out_root):
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)
    for src, dst in files:
        src = Path(src)
        if not src.exists():
            raise SystemExit(f"missing {src}")
        target = d / dst
        if target.exists() and target.stat().st_size == src.stat().st_size:
            continue
        target.unlink(missing_ok=True)
        try:
            os.link(src, target)          # multi-GB weights: link, don't copy
        except OSError:
            shutil.copy2(src, target)
    (d / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    shutil.copy2(HERE / "cards" / card, d / "README.md")
    for text_file in (d / "config.json", d / "README.md"):
        if SECRET.search(text_file.read_text()):
            raise SystemExit(f"credential-like string in {text_file}")
    stray = [p.name for p in d.iterdir()
             if p.name not in {dst for _, dst in files} | {"config.json", "README.md"}]
    if stray:
        raise SystemExit(f"unexpected files in {d}: {stray}")
    print(f"{name}: " + ", ".join(f"{p.name} {p.stat().st_size/1e6:.0f} MB"
                                  for p in sorted(d.iterdir())))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", type=Path, required=True,
                    help="directory with model.safetensors and artist_embeddings.safetensors")
    ap.add_argument("--classifier", type=Path, required=True, help="real-data classifier .pt")
    ap.add_argument("--synthetic-classifier", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "bundles")
    args = ap.parse_args()

    stage("jazz-pianist-style-generator",
          [(args.generator / "model.safetensors", "model.safetensors"),
           (args.generator / "artist_embeddings.safetensors", "artist_embeddings.safetensors")],
          GENERATOR_CONFIG, "generator.md", args.out)
    stage("jazz-pianist-style-classifier", [(args.classifier, "best.pt")],
          {**CLASSIFIER_CONFIG, "trained_on": "real PiJAMA-12 train split"},
          "classifier.md", args.out)
    stage("jazz-pianist-style-synthetic-classifier", [(args.synthetic_classifier, "best.pt")],
          {**CLASSIFIER_CONFIG, "trained_on": "generated continuations only (no real performances)"},
          "synthetic-classifier.md", args.out)


if __name__ == "__main__":
    main()
