#!/usr/bin/env python3
"""Evaluate a pianist classifier at chunk, clip, and track level.

Reproduces the classification numbers from the ISMIR 2026 paper "Learning
Jazz Pianist Style with Cross-Attention Conditioning": chunk-level top-1
accuracy plus track-level aggregation by majority vote (exact vote ties
broken by mean logit), average logits, and max confidence.

Example:
    python scripts/evaluate_classifier.py \
        --checkpoint checkpoints/pijama12_classifier.pt \
        --test-jsonl data/eval/pijama12_test_1024.jsonl \
        --artist-map data/eval/artist_to_id.json \
        --out results/pijama12_eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import jsonlines
import torch
import torch.nn.functional as F

from llama_pijama.evaluation import (
    compute_all_metrics,
    compute_track_level_metrics,
    compute_two_stage_track_metrics,
    load_model,
    run_inference,
)

logger = logging.getLogger("evaluate_classifier")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--test-jsonl", type=Path, required=True)
    ap.add_argument("--artist-map", type=Path, required=True,
                    help="artist_to_id.json used at training time")
    ap.add_argument("--out", type=Path, default=Path("evaluation_results.json"))
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Evaluate only the first N chunks (smoke tests)")
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    artist_to_id = json.loads(args.artist_map.read_text())

    # Per-chunk metadata: track and clip identity for aggregation
    track_ids, clip_ids, sample_meta = [], [], []
    with jsonlines.open(args.test_jsonl) as reader:
        for record in reader:
            meta = record.get("metadata", {})
            track_id = meta.get("track_id") or meta.get("midi_filepath") or f"track_{len(track_ids)}"
            track_ids.append(track_id)
            clip_ids.append(meta.get("midi_filepath") or track_id)
            sample_meta.append(meta)
            if args.max_chunks and len(track_ids) >= args.max_chunks:
                break
    logger.info(f"{len(track_ids)} chunks, {len(set(track_ids))} tracks, "
                f"{len(set(clip_ids))} clips")

    model, _ = load_model(str(args.checkpoint), model_name=args.model_name,
                          num_classes=len(artist_to_id), device=device)
    results = run_inference(model, str(args.test_jsonl), str(args.artist_map),
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            device=device, max_samples=args.max_chunks)

    chunk = compute_all_metrics(results["predictions"], results["true_labels"],
                                results["logits"], results["artist_to_id"],
                                results["id_to_artist"])
    clip = compute_track_level_metrics(results["predictions"], results["true_labels"],
                                       results["logits"], clip_ids, results["id_to_artist"])
    track = compute_track_level_metrics(results["predictions"], results["true_labels"],
                                        results["logits"], track_ids, results["id_to_artist"])
    two_stage = compute_two_stage_track_metrics(results["predictions"], results["true_labels"],
                                                results["logits"], clip_ids, track_ids,
                                                results["id_to_artist"])

    id_to_artist = results["id_to_artist"]
    per_sample = []
    for i, (pred, label) in enumerate(zip(results["predictions"], results["true_labels"])):
        meta = sample_meta[i] if i < len(sample_meta) else {}
        probs = F.softmax(torch.tensor(results["logits"][i]), dim=-1)
        top5_idx = torch.argsort(probs, descending=True)[:5]
        per_sample.append({
            "true_artist": id_to_artist.get(label, str(label)),
            "predicted_artist": id_to_artist.get(pred, str(pred)),
            "correct": bool(pred == label),
            "top5": [{"artist": id_to_artist.get(int(j), str(int(j))),
                      "score": float(probs[j])} for j in top5_idx],
            "track_id": meta.get("track_id", ""),
            "title": meta.get("title", ""),
            "logits": [round(float(v), 4) for v in results["logits"][i]],
        })

    out = {
        "chunk_level": chunk,
        "clip_level": {m: clip[f"track_level_{m}"]
                       for m in ("majority_vote", "avg_logits", "max_confidence")
                       if f"track_level_{m}" in clip},
        "track_level": {m: track[f"track_level_{m}"]
                        for m in ("majority_vote", "avg_logits", "max_confidence")
                        if f"track_level_{m}" in track},
        "track_level_two_stage": two_stage,
        "per_sample": per_sample,
        "artists": [id_to_artist[i] for i in range(len(id_to_artist))],
    }
    # metadata dicts are large and non-serializable-ish; summarize
    for level in ("clip_level", "track_level"):
        for m, d in out[level].items():
            if "metadata" in d:
                d["metadata_summary"] = f"{len(d.pop('metadata'))} groups"
    if "metadata" in out["track_level_two_stage"]:
        out["track_level_two_stage"]["metadata_summary"] = \
            f"{len(out['track_level_two_stage'].pop('metadata'))} tracks"

    logger.info(f"chunk top-1: {chunk['top1_accuracy']:.4f}")
    for m, d in out["track_level"].items():
        logger.info(f"track {m}: {d['accuracy']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
