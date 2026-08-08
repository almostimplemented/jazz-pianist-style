"""Metrics computation for PiJAMA classifier evaluation."""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def top_k_accuracy(predictions: List[int], true_labels: List[int], logits: List[List[float]], k: int = 1) -> float:
    """Compute top-k accuracy.

    Args:
        predictions: List of predicted class indices (only used if k=1)
        true_labels: List of true class indices
        logits: List of logit vectors
        k: k for top-k accuracy

    Returns:
        accuracy: Top-k accuracy as a float
    """
    if k == 1:
        correct = sum(p == t for p, t in zip(predictions, true_labels))
        return correct / len(true_labels) if true_labels else 0.0

    # For k > 1, use logits to get top-k predictions
    logits_array = np.array(logits)
    top_k_preds = np.argsort(logits_array, axis=1)[:, -k:]  # Top k predictions

    correct = 0
    for i, true_label in enumerate(true_labels):
        if true_label in top_k_preds[i]:
            correct += 1

    return correct / len(true_labels) if true_labels else 0.0


def per_class_accuracy(predictions: List[int], true_labels: List[int], id_to_artist: Dict[int, str]) -> Dict[str, float]:
    """Compute per-class (artist) accuracy.

    Args:
        predictions: List of predicted class indices
        true_labels: List of true class indices
        id_to_artist: Mapping from class ID to artist name

    Returns:
        accuracies: Dict mapping artist name to accuracy
    """
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for pred, true in zip(predictions, true_labels):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1

    accuracies = {}
    for class_id in class_total:
        artist = id_to_artist[class_id]
        accuracies[artist] = class_correct[class_id] / class_total[class_id]

    return accuracies


def confusion_matrix(predictions: List[int], true_labels: List[int], num_classes: int) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        predictions: List of predicted class indices
        true_labels: List of true class indices
        num_classes: Total number of classes

    Returns:
        cm: Confusion matrix of shape (num_classes, num_classes)
            cm[i, j] = number of times true class i was predicted as class j
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_labels, predictions):
        cm[true, pred] += 1
    return cm


def find_top_confusion_pairs(
    predictions: List[int],
    true_labels: List[int],
    id_to_artist: Dict[int, str],
    top_k: int = 10
) -> List[Tuple[str, str, int]]:
    """Find the most common confusion pairs (excluding correct predictions).

    Args:
        predictions: List of predicted class indices
        true_labels: List of true class indices
        id_to_artist: Mapping from class ID to artist name
        top_k: Number of top confusion pairs to return

    Returns:
        confusion_pairs: List of (true_artist, predicted_artist, count) tuples
    """
    confusion_counts = defaultdict(int)

    for true, pred in zip(true_labels, predictions):
        if true != pred:  # Only count misclassifications
            confusion_counts[(true, pred)] += 1

    # Sort by count and get top k
    sorted_pairs = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Convert to artist names
    confusion_pairs = [
        (id_to_artist[true_id], id_to_artist[pred_id], count)
        for (true_id, pred_id), count in sorted_pairs
    ]

    return confusion_pairs


def compute_all_metrics(
    predictions: List[int],
    true_labels: List[int],
    logits: List[List[float]],
    artist_to_id: Dict[str, int],
    id_to_artist: Dict[int, str],
) -> Dict:
    """Compute all standard metrics.

    Args:
        predictions: List of predicted class indices
        true_labels: List of true class indices
        logits: List of logit vectors
        artist_to_id: Artist name to ID mapping
        id_to_artist: ID to artist name mapping

    Returns:
        metrics: Dict containing all computed metrics
    """
    num_classes = len(artist_to_id)

    metrics = {
        "top1_accuracy": top_k_accuracy(predictions, true_labels, logits, k=1),
        "top2_accuracy": top_k_accuracy(predictions, true_labels, logits, k=2),
        "top3_accuracy": top_k_accuracy(predictions, true_labels, logits, k=3),
        "per_class_accuracy": per_class_accuracy(predictions, true_labels, id_to_artist),
        "confusion_matrix": confusion_matrix(predictions, true_labels, num_classes).tolist(),
        "top_confusion_pairs": find_top_confusion_pairs(predictions, true_labels, id_to_artist, top_k=20),
    }

    return metrics
