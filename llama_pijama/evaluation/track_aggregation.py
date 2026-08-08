"""Track-level aggregation for chunk-based predictions."""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


def aggregate_predictions_by_track(
    predictions: List[int],
    true_labels: List[int],
    logits: List[np.ndarray],
    track_ids: List[str],
    method: str = "majority_vote"
) -> Tuple[List[int], List[int], Dict]:
    """Aggregate chunk-level predictions to track-level.

    Args:
        predictions: Chunk-level predicted class indices
        true_labels: Chunk-level true class indices
        logits: Chunk-level logit vectors
        track_ids: Track identifier for each chunk (e.g., midi_filepath)
        method: Aggregation method ("majority_vote", "avg_logits", "max_confidence")

    Returns:
        track_predictions: Track-level predictions
        track_labels: Track-level true labels
        track_metadata: Dict with aggregation details
    """
    # Group chunks by track
    track_chunks = defaultdict(list)
    for i, track_id in enumerate(track_ids):
        track_chunks[track_id].append(i)

    track_predictions = []
    track_labels = []
    track_metadata = {}

    for track_id, chunk_indices in track_chunks.items():
        # Get all predictions/labels for this track
        track_preds = [predictions[i] for i in chunk_indices]
        track_true = [true_labels[i] for i in chunk_indices]
        track_logits_list = [logits[i] for i in chunk_indices]

        # Verify all chunks have same true label (they should)
        assert len(set(track_true)) == 1, f"Mixed labels in track {track_id}: {track_true}"
        track_label = track_true[0]

        # Aggregate predictions based on method
        if method == "majority_vote":
            # Most common prediction; exact ties broken by mean logit
            # among the tied classes (Counter insertion order is not a
            # meaningful tie-break)
            from collections import Counter
            vote_counts = Counter(track_preds)
            top_count = vote_counts.most_common(1)[0][1]
            tied = [c for c, n in vote_counts.items() if n == top_count]
            if len(tied) > 1:
                mean_logits = np.mean(track_logits_list, axis=0)
                track_pred = max(tied, key=lambda c: mean_logits[c])
            else:
                track_pred = tied[0]
            confidence = vote_counts[track_pred] / len(track_preds)

        elif method == "avg_logits":
            # Average logits, then argmax
            avg_logits = np.mean(track_logits_list, axis=0)
            track_pred = np.argmax(avg_logits)
            confidence = np.exp(avg_logits[track_pred]) / np.sum(np.exp(avg_logits))

        elif method == "max_confidence":
            # Take prediction with highest confidence
            max_conf = -float('inf')
            track_pred = track_preds[0]
            for i, pred in enumerate(track_preds):
                # Softmax to get confidence
                probs = np.exp(track_logits_list[i]) / np.sum(np.exp(track_logits_list[i]))
                if probs[pred] > max_conf:
                    max_conf = probs[pred]
                    track_pred = pred
            confidence = max_conf
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        track_predictions.append(track_pred)
        track_labels.append(track_label)

        # Store metadata about aggregation
        track_metadata[track_id] = {
            "num_chunks": len(chunk_indices),
            "chunk_predictions": track_preds,
            "track_prediction": track_pred,
            "track_label": track_label,
            "confidence": float(confidence),
            "chunk_agreement": sum(p == track_pred for p in track_preds) / len(track_preds)
        }

    return track_predictions, track_labels, track_metadata


def compute_two_stage_track_metrics(
    predictions: List[int],
    true_labels: List[int],
    logits: List[np.ndarray],
    clip_ids: List[str],
    track_ids: List[str],
    id_to_artist: Dict[int, str]
) -> Dict:
    """Two-stage aggregation: chunks → clip probabilities → track prediction.

    Matches the methodology from Cheston et al. (2025): softmax per chunk,
    average probabilities within each clip, then average clip probabilities
    within each track. Each clip gets equal weight regardless of chunk count.

    Returns dict with track-level accuracy and per-artist breakdown.
    """
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # Stage 1: aggregate chunks → clip-level probabilities
    clip_chunks = defaultdict(list)
    clip_labels = {}
    clip_track_map = {}
    for i, clip_id in enumerate(clip_ids):
        clip_chunks[clip_id].append(i)
        clip_labels[clip_id] = true_labels[i]
        clip_track_map[clip_id] = track_ids[i]

    clip_probs = {}
    for clip_id, chunk_indices in clip_chunks.items():
        # Softmax per chunk, then average across chunks within this clip
        chunk_probs = [_softmax(logits[i]) for i in chunk_indices]
        clip_probs[clip_id] = np.mean(chunk_probs, axis=0)

    # Stage 2: aggregate clip probabilities → track prediction
    track_clips = defaultdict(list)
    track_labels_map = {}
    for clip_id, track_id in clip_track_map.items():
        track_clips[track_id].append(clip_id)
        track_labels_map[track_id] = clip_labels[clip_id]

    track_preds = []
    track_labels_list = []
    track_meta = {}

    for track_id, clip_id_list in track_clips.items():
        # Average clip-level probabilities (equal weight per clip)
        avg_probs = np.mean([clip_probs[c] for c in clip_id_list], axis=0)
        track_pred = int(np.argmax(avg_probs))
        track_label = track_labels_map[track_id]

        track_preds.append(track_pred)
        track_labels_list.append(track_label)
        track_meta[track_id] = {
            "num_clips": len(clip_id_list),
            "num_chunks": sum(len(clip_chunks[c]) for c in clip_id_list),
            "track_prediction": track_pred,
            "track_label": track_label,
            "confidence": float(avg_probs[track_pred]),
        }

    track_acc = sum(p == t for p, t in zip(track_preds, track_labels_list)) / len(track_preds)

    # Per-artist accuracy
    artist_correct = defaultdict(int)
    artist_total = defaultdict(int)
    for pred, true_label in zip(track_preds, track_labels_list):
        artist = id_to_artist[true_label]
        artist_total[artist] += 1
        if pred == true_label:
            artist_correct[artist] += 1

    per_artist_acc = {
        artist: artist_correct[artist] / artist_total[artist]
        for artist in artist_total
    }

    return {
        "accuracy": track_acc,
        "total_tracks": len(track_preds),
        "total_clips": len(clip_probs),
        "per_artist_accuracy": per_artist_acc,
        "metadata": track_meta,
    }


def compute_track_level_metrics(
    predictions: List[int],
    true_labels: List[int],
    logits: List[np.ndarray],
    track_ids: List[str],
    id_to_artist: Dict[int, str]
) -> Dict:
    """Compute both chunk-level and track-level metrics.

    Returns dict with both chunk and track level accuracies.
    """
    # Chunk-level accuracy
    chunk_acc = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)

    results = {
        "chunk_level": {
            "accuracy": chunk_acc,
            "total_chunks": len(predictions)
        }
    }

    # Track-level accuracies for different methods
    for method in ["majority_vote", "avg_logits", "max_confidence"]:
        track_preds, track_labels, track_meta = aggregate_predictions_by_track(
            predictions, true_labels, logits, track_ids, method=method
        )

        track_acc = sum(p == t for p, t in zip(track_preds, track_labels)) / len(track_preds)

        # Per-artist track accuracy
        artist_correct = defaultdict(int)
        artist_total = defaultdict(int)
        for pred, true_label in zip(track_preds, track_labels):
            artist = id_to_artist[true_label]
            artist_total[artist] += 1
            if pred == true_label:
                artist_correct[artist] += 1

        per_artist_acc = {
            artist: artist_correct[artist] / artist_total[artist]
            for artist in artist_total
        }

        results[f"track_level_{method}"] = {
            "accuracy": track_acc,
            "total_tracks": len(track_preds),
            "per_artist_accuracy": per_artist_acc,
            "metadata": track_meta
        }

    return results