"""Inference utilities for evaluating PiJAMA classifier models."""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm

from llama_pijama.training.pijama_classifier import JsonlClassificationDataset
from ariautils.tokenizer import AbsTokenizer


def load_model(checkpoint_path: str, model_name: str = "medium", num_classes: int = 30, device: str = "cuda"):
    """Load a trained classifier model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        model_name: Aria model size (small, medium, large)
        num_classes: Number of classification classes
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
        model_config: Model configuration
    """
    from aria.config import load_model_config
    from aria.model import TransformerCL, ModelConfig

    tokenizer = AbsTokenizer()
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.class_size = num_classes
    model_config.max_seq_len = 1024

    model = TransformerCL(model_config)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, model_config


def run_inference(
    model,
    data_path: str,
    artist_to_id_path: str,
    batch_size: int = 1,
    num_workers: int = 2,
    device: str = "cuda",
    max_samples: int = None,
):
    """Run inference on a dataset and return predictions with logits.

    Args:
        model: Trained model in eval mode
        data_path: Path to JSONL dataset
        artist_to_id_path: Path to artist_to_id.json mapping
        batch_size: Batch size for inference
        num_workers: Number of dataloader workers
        device: Device for inference

    Returns:
        results: Dict with keys:
            - predictions: List of predicted class indices
            - true_labels: List of true class indices
            - logits: List of logit vectors (for top-k and similarity analysis)
            - artist_to_id: Artist name to ID mapping
            - id_to_artist: ID to artist name mapping
    """
    with open(artist_to_id_path, "r") as f:
        artist_to_id = json.load(f)

    id_to_artist = {v: k for k, v in artist_to_id.items()}

    # Load dataset
    dataset = JsonlClassificationDataset(
        data_path,
        artist_to_id,
        max_seq_len=1024
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    predictions = []
    true_labels = []
    all_logits = []

    with torch.no_grad():
        for seqs, eos_pos, labels in tqdm(dataloader, desc="Running inference"):
            if max_samples is not None and len(predictions) >= max_samples:
                break
            seqs = seqs.to(device)
            eos_pos = eos_pos.to(device)

            # Forward pass
            logits = model(seqs)
            # Get logits at EOS position
            logits_at_eos = logits[torch.arange(logits.shape[0], device=device), eos_pos]

            # Store results
            preds = logits_at_eos.argmax(dim=-1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.tolist())
            all_logits.extend(logits_at_eos.cpu().tolist())

    return {
        "predictions": predictions,
        "true_labels": true_labels,
        "logits": all_logits,
        "artist_to_id": artist_to_id,
        "id_to_artist": id_to_artist,
    }
