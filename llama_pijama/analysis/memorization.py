"""Memorization and plagiarism detection for generated music.

Compares generated sequences against training data using three complementary
methods at varying granularities. The primary diagnostic is the overlap-vs-n
decay curve: a memorizing model shows overlap persisting at larger n values.

Methods:
    1. Pitch n-gram overlap: fraction of generated pitch n-grams found in training
    2. Interval n-gram overlap: same, but on pitch intervals (transposition-invariant)
    3. Piano roll Jaccard: max windowed Jaccard similarity on binary piano rolls
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Instrument names used in Aria token tuples
_INSTRUMENTS = {
    "piano", "organ", "guitar", "bass", "strings",
    "ensemble", "brass", "reed", "pipe", "synth_lead",
}


@dataclass
class NoteEvent:
    """Single note with pitch, timing, and dynamics."""
    pitch: int        # MIDI 0-127
    onset_ms: int     # absolute start time
    duration_ms: int
    velocity: int


# ---------------------------------------------------------------------------
# Note extraction from two token formats
# ---------------------------------------------------------------------------

def extract_notes_from_tuples(seq: List) -> List[NoteEvent]:
    """Extract notes from a token-tuple sequence (training JSONL format).

    AbsTokenizer semantics (matches scripts/plot_characteristic_pianoroll.py,
    the reference reconstruction): token order is note -> onset -> dur per
    note event; "onset" is ABSOLUTE within the current 5000 ms segment, and
    the string token "<T>" advances the segment base by 5000 ms. (The old
    implementation treated onsets as relative shifts and ignored "<T>",
    corrupting all timing — see GOTCHAS.md 2026-06-09.)
    """
    abs_step_ms = 5000
    notes: List[NoteEvent] = []
    time_base = 0
    pending: Optional[NoteEvent] = None
    pending_has_onset = False

    def flush():
        nonlocal pending, pending_has_onset
        if pending is not None:
            notes.append(pending)
        pending = None
        pending_has_onset = False

    for tok in seq:
        if tok == "<T>":
            time_base += abs_step_ms
            continue
        if isinstance(tok, str):
            continue
        if not isinstance(tok, (list, tuple)) or len(tok) < 2:
            continue

        tok_type = tok[0]

        if tok_type in _INSTRUMENTS:
            flush()
            pending = NoteEvent(
                pitch=tok[1],
                onset_ms=time_base,  # placeholder until the onset token
                duration_ms=100,
                velocity=(tok[2] if len(tok) > 2 else 64) or 64,
            )
        elif tok_type == "onset":
            if pending is not None and not pending_has_onset:
                pending.onset_ms = time_base + tok[1]
                pending_has_onset = True
        elif tok_type == "dur":
            if pending is not None:
                pending.duration_ms = tok[1]
                flush()

    flush()
    return notes


def load_training_sequences(
    jsonl_path: str,
    max_sequences: Optional[int] = None,
) -> Dict[str, List[List[NoteEvent]]]:
    """Load training data from JSONL and extract note sequences by artist."""
    sequences: Dict[str, List[List[NoteEvent]]] = defaultdict(list)
    count = 0

    with jsonlines.open(jsonl_path) as reader:
        for record in tqdm(reader, desc="Loading training data"):
            artist = record.get("metadata", {}).get("artist", "unknown")
            notes = extract_notes_from_tuples(record["seq"])
            if notes:
                sequences[artist].append(notes)
            count += 1
            if max_sequences and count >= max_sequences:
                break

    logger.info(
        f"Loaded {count} sequences from {len(sequences)} artists "
        f"({sum(len(v) for v in sequences.values())} with notes)"
    )
    return dict(sequences)


def _pitches(notes: List[NoteEvent]) -> Tuple[int, ...]:
    return tuple(n.pitch for n in notes)


def _intervals(notes: List[NoteEvent]) -> Tuple[int, ...]:
    pitches = [n.pitch for n in notes]
    return tuple(b - a for a, b in zip(pitches, pitches[1:]))


def _ngrams(seq: Tuple[int, ...], n: int) -> set:
    """Return set of all n-grams from a sequence."""
    if len(seq) < n:
        return set()
    return {seq[i:i + n] for i in range(len(seq) - n + 1)}


def _build_piano_roll(
    notes: List[NoteEvent],
    resolution_ms: int = 50,
) -> np.ndarray:
    """Build a binary piano roll (128 x time_bins) from note events."""
    if not notes:
        return np.zeros((128, 0), dtype=bool)

    max_end = max(n.onset_ms + n.duration_ms for n in notes)
    num_bins = max(1, int(np.ceil(max_end / resolution_ms)))
    roll = np.zeros((128, num_bins), dtype=bool)

    for n in notes:
        start_bin = n.onset_ms // resolution_ms
        end_bin = min(num_bins, int(np.ceil((n.onset_ms + n.duration_ms) / resolution_ms)))
        if 0 <= n.pitch < 128 and start_bin < num_bins:
            roll[n.pitch, start_bin:end_bin] = True

    return roll


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class MemorizationAnalyzer:
    """Compare generated sequences against training data for memorization.

    Pre-indexes training data for efficient repeated queries.
    """

    def __init__(
        self,
        training_sequences: Dict[str, List[List[NoteEvent]]],
        resolution_ms: int = 50,
        verbose: bool = False,
    ):
        self.resolution_ms = resolution_ms
        self.verbose = verbose

        # Store raw note sequences by artist
        self._training = training_sequences

        # Extract pitch and interval sequences per artist
        self._pitch_seqs: Dict[str, List[Tuple[int, ...]]] = {}
        self._interval_seqs: Dict[str, List[Tuple[int, ...]]] = {}
        for artist, seqs in training_sequences.items():
            self._pitch_seqs[artist] = [_pitches(s) for s in seqs]
            self._interval_seqs[artist] = [_intervals(s) for s in seqs if len(s) >= 2]

        # Lazy n-gram caches: artist -> n -> set
        self._pitch_ngram_cache: Dict[str, Dict[int, set]] = defaultdict(dict)
        self._interval_ngram_cache: Dict[str, Dict[int, set]] = defaultdict(dict)
        # "all" key for union across artists
        self._pitch_ngram_cache_all: Dict[int, set] = {}
        self._interval_ngram_cache_all: Dict[int, set] = {}

        # Pre-build piano rolls
        self._piano_rolls: Dict[str, List[np.ndarray]] = {}
        for artist, seqs in training_sequences.items():
            self._piano_rolls[artist] = [
                _build_piano_roll(s, resolution_ms) for s in seqs
            ]

        if verbose:
            total = sum(len(v) for v in training_sequences.values())
            logger.info(f"Indexed {total} training sequences from {len(training_sequences)} artists")

    def _get_pitch_ngrams(self, n: int, artist: Optional[str] = None) -> set:
        """Get or build the pitch n-gram set for an artist (or all)."""
        if artist is not None:
            if n not in self._pitch_ngram_cache[artist]:
                result = set()
                for seq in self._pitch_seqs.get(artist, []):
                    result |= _ngrams(seq, n)
                self._pitch_ngram_cache[artist][n] = result
            return self._pitch_ngram_cache[artist][n]
        else:
            if n not in self._pitch_ngram_cache_all:
                result = set()
                for artist_name in self._pitch_seqs:
                    result |= self._get_pitch_ngrams(n, artist_name)
                self._pitch_ngram_cache_all[n] = result
            return self._pitch_ngram_cache_all[n]

    def _get_interval_ngrams(self, n: int, artist: Optional[str] = None) -> set:
        """Get or build the interval n-gram set for an artist (or all)."""
        if artist is not None:
            if n not in self._interval_ngram_cache[artist]:
                result = set()
                for seq in self._interval_seqs.get(artist, []):
                    result |= _ngrams(seq, n)
                self._interval_ngram_cache[artist][n] = result
            return self._interval_ngram_cache[artist][n]
        else:
            if n not in self._interval_ngram_cache_all:
                result = set()
                for artist_name in self._interval_seqs:
                    result |= self._get_interval_ngrams(n, artist_name)
                self._interval_ngram_cache_all[n] = result
            return self._interval_ngram_cache_all[n]

    # -----------------------------------------------------------------------
    # Single-sequence methods
    # -----------------------------------------------------------------------

    def pitch_ngram_overlap(
        self,
        generated: List[NoteEvent],
        n_values: Optional[List[int]] = None,
        reference_artist: Optional[str] = None,
    ) -> Dict[int, float]:
        """Fraction of generated pitch n-grams found in reference data."""
        if n_values is None:
            n_values = list(range(3, 51))

        gen_pitches = _pitches(generated)
        result = {}
        for n in n_values:
            gen_ng = _ngrams(gen_pitches, n)
            if not gen_ng:
                result[n] = 0.0
                continue
            ref_ng = self._get_pitch_ngrams(n, reference_artist)
            result[n] = len(gen_ng & ref_ng) / len(gen_ng)
        return result

    def interval_ngram_overlap(
        self,
        generated: List[NoteEvent],
        n_values: Optional[List[int]] = None,
        reference_artist: Optional[str] = None,
    ) -> Dict[int, float]:
        """Fraction of generated interval n-grams found in reference data."""
        if n_values is None:
            n_values = list(range(3, 51))

        gen_intervals = _intervals(generated)
        result = {}
        for n in n_values:
            gen_ng = _ngrams(gen_intervals, n)
            if not gen_ng:
                result[n] = 0.0
                continue
            ref_ng = self._get_interval_ngrams(n, reference_artist)
            result[n] = len(gen_ng & ref_ng) / len(gen_ng)
        return result

    def piano_roll_jaccard(
        self,
        generated: List[NoteEvent],
        window_sizes_ms: Optional[List[int]] = None,
        reference_artist: Optional[str] = None,
        max_ref_rolls: int = 200,
    ) -> Dict[int, float]:
        """Max Jaccard similarity of piano roll windows against reference.

        Args:
            max_ref_rolls: Subsample reference rolls if there are more than this.
        """
        import random as _rng

        if window_sizes_ms is None:
            window_sizes_ms = [500, 1000, 2000, 4000, 8000]

        gen_roll = _build_piano_roll(generated, self.resolution_ms)
        if gen_roll.shape[1] == 0:
            return {w: 0.0 for w in window_sizes_ms}

        # Collect reference rolls (subsample if too many)
        if reference_artist is not None:
            ref_rolls = self._piano_rolls.get(reference_artist, [])
        else:
            ref_rolls = [r for rolls in self._piano_rolls.values() for r in rolls]

        if len(ref_rolls) > max_ref_rolls:
            ref_rolls = _rng.sample(ref_rolls, max_ref_rolls)

        result = {}
        for window_ms in window_sizes_ms:
            w_bins = max(1, window_ms // self.resolution_ms)
            max_j = 0.0

            # Extract all generated windows as flat vectors. Zero-pad rolls
            # shorter than the window so every window has the same D as the
            # reference windows (the ref side skips short rolls instead).
            step = max(1, w_bins // 2)
            roll = gen_roll
            if roll.shape[1] < w_bins:
                pad = np.zeros((roll.shape[0], w_bins - roll.shape[1]), dtype=roll.dtype)
                roll = np.concatenate([roll, pad], axis=1)
            gen_starts = list(range(0, roll.shape[1] - w_bins + 1, step))
            if not gen_starts:
                result[window_ms] = 0.0
                continue

            gen_windows = np.stack([
                roll[:, s:s + w_bins].ravel() for s in gen_starts
            ])  # shape: (num_gen_windows, 128 * w_bins)

            # Skip empty windows
            gen_active = gen_windows.sum(axis=1) > 0
            if not gen_active.any():
                result[window_ms] = 0.0
                continue
            gen_windows = gen_windows[gen_active]

            # Flatten generated windows to uint8 for faster ops
            gen_windows = gen_windows.astype(np.uint8)

            # Compare against reference rolls in vectorized batches
            for ref_roll in ref_rolls:
                if ref_roll.shape[1] < w_bins:
                    continue

                ref_starts = list(range(0, ref_roll.shape[1] - w_bins + 1, step))
                if not ref_starts:
                    continue

                ref_windows = np.stack([
                    ref_roll[:, s:s + w_bins].ravel() for s in ref_starts
                ]).astype(np.uint8)  # (num_ref_windows, D)

                # Batch Jaccard: each gen window vs all ref windows
                for g_w in gen_windows:
                    inter = (g_w & ref_windows).sum(axis=1)
                    union = (g_w | ref_windows).sum(axis=1)
                    valid = union > 0
                    if valid.any():
                        j = float((inter[valid] / union[valid]).max())
                        if j > max_j:
                            max_j = j
                            if max_j > 0.99:
                                break
                if max_j > 0.99:
                    break

            result[window_ms] = max_j

        return result

    # -----------------------------------------------------------------------
    # Batch analysis
    # -----------------------------------------------------------------------

    def analyze_batch(
        self,
        generated_sequences: Dict[str, List[List[NoteEvent]]],
        n_values: Optional[List[int]] = None,
        window_sizes_ms: Optional[List[int]] = None,
        cross_artist: bool = True,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run selected methods on a batch of generated sequences.

        Args:
            methods: Which methods to run. Default: all three.
                Options: "pitch_ngram", "interval_ngram", "piano_roll_jaccard"

        Returns nested dict:
            {method: {condition: {n_or_window: {mean, std, values}}}}

        Conditions are "same_artist" and optionally "cross_artist".
        """
        if n_values is None:
            n_values = list(range(3, 51))
        if window_sizes_ms is None:
            window_sizes_ms = [500, 1000, 2000, 4000, 8000]

        all_methods = {
            "pitch_ngram": (self.pitch_ngram_overlap, {"n_values": n_values}),
            "interval_ngram": (self.interval_ngram_overlap, {"n_values": n_values}),
            "piano_roll_jaccard": (self.piano_roll_jaccard, {"window_sizes_ms": window_sizes_ms}),
        }

        if methods is not None:
            all_methods = {k: v for k, v in all_methods.items() if k in methods}

        results: Dict[str, Dict[str, Dict]] = {m: {} for m in all_methods}
        training_artists = set(self._training.keys())

        total_samples = sum(len(seqs) for seqs in generated_sequences.values())
        pbar = tqdm(total=total_samples * len(all_methods) * (2 if cross_artist else 1),
                    desc="Analyzing memorization", disable=not self.verbose)

        import random

        for method_name, (method_fn, method_kwargs) in all_methods.items():
            same_artist_scores: Dict[int, List[float]] = defaultdict(list)
            cross_artist_scores: Dict[int, List[float]] = defaultdict(list)
            per_artist_same: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

            for artist, seqs in generated_sequences.items():
                # Pre-pick a random other artist for cross-artist comparison
                # (one per artist, not per sample — cheap and stable enough)
                other_artists = list(training_artists - {artist})

                for seq in seqs:
                    # Same-artist comparison
                    ref = artist if artist in training_artists else None
                    scores = method_fn(seq, reference_artist=ref, **method_kwargs)
                    for k, v in scores.items():
                        same_artist_scores[k].append(v)
                        per_artist_same[artist][k].append(v)
                    pbar.update(1)

                    # Cross-artist: compare against one randomly chosen other artist
                    if cross_artist and other_artists:
                        other = random.choice(other_artists)
                        cs = method_fn(seq, reference_artist=other, **method_kwargs)
                        for k, v in cs.items():
                            cross_artist_scores[k].append(v)
                    pbar.update(1)

            results[method_name]["same_artist"] = _summarize_scores(same_artist_scores)
            results[method_name]["per_artist"] = {
                artist: _summarize_scores(scores)
                for artist, scores in per_artist_same.items()
            }
            if cross_artist:
                results[method_name]["cross_artist"] = _summarize_scores(cross_artist_scores)

        pbar.close()
        return results

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot_overlap_curves(
        self,
        results: Dict[str, Any],
        output_dir: Optional[Path] = None,
        baseline_results: Optional[Dict[str, Any]] = None,
        title_prefix: str = "",
    ) -> Dict[str, "matplotlib.figure.Figure"]:
        """Plot overlap-vs-n/window decay curves.

        Returns dict of figures keyed by method name.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures = {}

        plot_configs = {
            "pitch_ngram": ("Pitch N-gram Overlap", "n", "Overlap Fraction"),
            "interval_ngram": ("Interval N-gram Overlap", "n", "Overlap Fraction"),
            "piano_roll_jaccard": ("Piano Roll Jaccard", "Window Size (ms)", "Max Jaccard"),
        }

        for method, (title, xlabel, ylabel) in plot_configs.items():
            if method not in results:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            for condition, style in [("same_artist", "-"), ("cross_artist", "--")]:
                if condition not in results[method]:
                    continue
                data = results[method][condition]
                keys = sorted(data["mean"].keys(), key=lambda x: int(x))
                xs = [int(k) for k in keys]
                means = [data["mean"][k] for k in keys]
                stds = [data["std"][k] for k in keys]

                label = condition.replace("_", " ").title()
                ax.plot(xs, means, style, label=label, linewidth=2)
                ax.fill_between(
                    xs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2,
                )

            # Overlay baseline if provided
            if baseline_results and method in baseline_results:
                for condition, style in [("same_artist", ":")]:
                    if condition not in baseline_results[method]:
                        continue
                    data = baseline_results[method][condition]
                    keys = sorted(data["mean"].keys(), key=lambda x: int(x))
                    xs = [int(k) for k in keys]
                    means = [data["mean"][k] for k in keys]
                    ax.plot(xs, means, style, label="Baseline", linewidth=2, color="gray")

            full_title = f"{title_prefix}{title}" if title_prefix else title
            ax.set_title(full_title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.05, 1.05)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            figures[method] = fig

            if output_dir:
                from .visualization import save_figure
                save_figure(fig, Path(output_dir) / f"{method}_curve.png")

        return figures


def _summarize_scores(
    scores: Dict[int, List[float]],
) -> Dict[str, Dict]:
    """Summarize per-key score lists into mean/std/values."""
    summary: Dict[str, Dict] = {"mean": {}, "std": {}, "values": {}}
    for k, vals in sorted(scores.items()):
        key = str(k)
        summary["mean"][key] = float(np.mean(vals)) if vals else 0.0
        summary["std"][key] = float(np.std(vals)) if vals else 0.0
        summary["values"][key] = vals
    return summary
