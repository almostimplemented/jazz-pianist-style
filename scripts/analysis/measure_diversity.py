#!/usr/bin/env python3
"""Within-condition diversity measurement for synthetic generations.

Computes pairwise n-gram Jaccard similarity between same-artist generations
(the "within-artist" distribution) and between different-artist generations
(the "between-artist" distribution). The healthy result is that within-artist
is clearly higher than between-artist (same style shares more structure)
but nowhere near 1.0 (not collapsed to near-duplicates).

Both distributions are computed on the same set of generated sequences.
Typical interpretation:
    within ~ between    -> generator captures no artist-specific style
    within > between,
      within << 1.0     -> healthy; generator has learned style variation
    within ~ 1.0         -> COLLAPSE; generator produces near-duplicates

Usage (the paper reports n=4; n=6 and n=8 as sensitivity checks):
    python scripts/analysis/measure_diversity.py \
        --jsonl data/synthetic/train.jsonl \
        --n 4 \
        --max-pairs-per-artist 500 \
        --output-json results/diversity_synth_n4.json

The `--jsonl` file should be in classifier format (seq + metadata.artist).
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import jsonlines

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llama_pijama.analysis.memorization import (
    extract_notes_from_tuples,
    _pitches,
    _intervals,
    _ngrams,
)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_by_artist(jsonl_path: Path, n: int) -> Dict[str, List[Tuple[set, set]]]:
    """Load sequences, group by artist, return per-sequence (pitch_ngrams, interval_ngrams).

    Sequences with no valid notes are dropped.
    """
    by_artist: Dict[str, List[Tuple[set, set]]] = defaultdict(list)
    with jsonlines.open(jsonl_path) as reader:
        for rec in reader:
            artist = rec.get("metadata", {}).get("artist")
            seq = rec.get("seq")
            if not artist or not seq:
                continue
            notes = extract_notes_from_tuples(seq)
            if len(notes) < n + 1:
                continue
            pitch_ng = _ngrams(_pitches(notes), n)
            int_ng = _ngrams(_intervals(notes), n) if len(notes) >= 2 else set()
            if not pitch_ng:
                continue
            by_artist[artist].append((pitch_ng, int_ng))
    return by_artist


def summarize(values: List[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
    import statistics
    values_sorted = sorted(values)
    q = lambda p: values_sorted[int(p * (len(values_sorted) - 1))]
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p25": q(0.25),
        "p75": q(0.75),
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="classifier-format jsonl (seq + metadata.artist)")
    p.add_argument("--n", type=int, default=4, help="n-gram size (default: 4, as in the paper)")
    p.add_argument("--max-pairs-per-artist", type=int, default=500,
                   help="max within-artist pairs to sample per artist (default: 500)")
    p.add_argument("--max-between-pairs", type=int, default=5000,
                   help="max between-artist pairs to sample across all artists (default: 5000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    random.seed(args.seed)
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    print(f"Loading {jsonl_path}")
    by_artist = load_by_artist(jsonl_path, args.n)
    total_seqs = sum(len(v) for v in by_artist.values())
    print(f"  {total_seqs} sequences across {len(by_artist)} artists (n={args.n})")
    for artist in sorted(by_artist):
        print(f"  {artist:<25} {len(by_artist[artist]):>5} seqs")

    # Within-artist pairs: per artist, sample random pairs and compute jaccard
    print(f"\nComputing within-artist pairwise jaccard...")
    per_artist_within_pitch: Dict[str, List[float]] = {}
    per_artist_within_interval: Dict[str, List[float]] = {}
    all_within_pitch: List[float] = []
    all_within_interval: List[float] = []

    for artist, seqs in by_artist.items():
        k = len(seqs)
        if k < 2:
            per_artist_within_pitch[artist] = []
            per_artist_within_interval[artist] = []
            continue
        # Generate all pairs if small, else sample
        all_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        random.shuffle(all_pairs)
        pairs = all_pairs[: args.max_pairs_per_artist]
        pitch_values = []
        interval_values = []
        for i, j in pairs:
            pitch_values.append(jaccard(seqs[i][0], seqs[j][0]))
            interval_values.append(jaccard(seqs[i][1], seqs[j][1]))
        per_artist_within_pitch[artist] = pitch_values
        per_artist_within_interval[artist] = interval_values
        all_within_pitch.extend(pitch_values)
        all_within_interval.extend(interval_values)

    # Between-artist pairs: sample random (i,j) where artist(i) != artist(j)
    print("Computing between-artist pairwise jaccard...")
    flat: List[Tuple[str, set, set]] = []
    for artist, seqs in by_artist.items():
        for pitch_ng, int_ng in seqs:
            flat.append((artist, pitch_ng, int_ng))

    between_pitch: List[float] = []
    between_interval: List[float] = []
    attempts = 0
    target = args.max_between_pairs
    while len(between_pitch) < target and attempts < target * 4:
        attempts += 1
        a = random.choice(flat)
        b = random.choice(flat)
        if a[0] == b[0]:
            continue
        between_pitch.append(jaccard(a[1], b[1]))
        between_interval.append(jaccard(a[2], b[2]))

    # Summaries
    results = {
        "config": {
            "jsonl": str(jsonl_path),
            "n": args.n,
            "max_pairs_per_artist": args.max_pairs_per_artist,
            "max_between_pairs": args.max_between_pairs,
            "seed": args.seed,
        },
        "within_artist_pitch": {
            "aggregate": summarize(all_within_pitch),
            "per_artist": {a: summarize(v) for a, v in per_artist_within_pitch.items()},
        },
        "within_artist_interval": {
            "aggregate": summarize(all_within_interval),
            "per_artist": {a: summarize(v) for a, v in per_artist_within_interval.items()},
        },
        "between_artist_pitch": summarize(between_pitch),
        "between_artist_interval": summarize(between_interval),
    }

    # Print compact summary
    print("\n" + "=" * 60)
    print(f"Diversity summary (n={args.n} pitch / interval n-grams, Jaccard)")
    print("=" * 60)
    wp = results["within_artist_pitch"]["aggregate"]
    bp = results["between_artist_pitch"]
    wi = results["within_artist_interval"]["aggregate"]
    bi = results["between_artist_interval"]
    print(f"Pitch n-grams:")
    print(f"  Within-artist:  mean={wp['mean']:.4f}  median={wp['median']:.4f}  [p25={wp['p25']:.4f}, p75={wp['p75']:.4f}]  n={wp['n']}")
    print(f"  Between-artist: mean={bp['mean']:.4f}  median={bp['median']:.4f}  [p25={bp['p25']:.4f}, p75={bp['p75']:.4f}]  n={bp['n']}")
    ratio_pitch = wp["mean"] / bp["mean"] if bp["mean"] > 0 else float("inf")
    print(f"  Within/Between ratio: {ratio_pitch:.2f}x")
    print(f"\nInterval n-grams:")
    print(f"  Within-artist:  mean={wi['mean']:.4f}  median={wi['median']:.4f}  [p25={wi['p25']:.4f}, p75={wi['p75']:.4f}]  n={wi['n']}")
    print(f"  Between-artist: mean={bi['mean']:.4f}  median={bi['median']:.4f}  [p25={bi['p25']:.4f}, p75={bi['p75']:.4f}]  n={bi['n']}")
    ratio_int = wi["mean"] / bi["mean"] if bi["mean"] > 0 else float("inf")
    print(f"  Within/Between ratio: {ratio_int:.2f}x")

    print(f"\nPer-artist within-artist pitch Jaccard (mean):")
    per_artist = results["within_artist_pitch"]["per_artist"]
    for a in sorted(per_artist):
        s = per_artist[a]
        if s["n"] > 0:
            print(f"  {a:<25} mean={s['mean']:.4f}  median={s['median']:.4f}  n_pairs={s['n']}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved full results to {args.output_json}")


if __name__ == "__main__":
    main()
