#!/usr/bin/env python3
"""Locate the most and least characteristic regions within a performance.

Repurposes the pianist classifier as an interpretability tool, following
Section 7 of the ISMIR 2026 paper "Learning Jazz Pianist Style with
Cross-Attention Conditioning". A 1024-token window slides along a full
performance; at each position we take the classifier's logit margin for the
true artist (its logit minus the strongest competing artist), convert those
margins to a within-track z-score, and smooth them. Peaks mark passages the
classifier finds most distinctive of the artist; troughs mark the least.

Outputs per track: the score curve as .npz, a summary JSON, and optionally
MIDI excerpts of the peak and trough regions.

Example:
    python scripts/characteristic_regions.py \
        --classifier checkpoints/pijama12_classifier.pt \
        --test-jsonl data/eval/pijama12_test_1024.jsonl \
        --artist-map data/eval/artist_to_id.json \
        --out-dir results/characteristic_regions --save-excerpts
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from ariautils.tokenizer import AbsTokenizer

from llama_pijama.evaluation import load_model as load_classifier

logger = logging.getLogger("characteristic_regions")

WINDOW_SIZE = 1024


def slugify(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", s).strip("_")[:80]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--test-jsonl", type=Path, required=True,
                    help="Chunked JSONL; chunks are reassembled per track")
    ap.add_argument("--artist-map", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/characteristic_regions"))
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--artist", default=None, help="Restrict to one artist")
    ap.add_argument("--track-contains", default=None,
                    help="Restrict to tracks whose title contains this string")
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--save-excerpts", action="store_true",
                    help="Write MIDI for the peak and trough regions")
    ap.add_argument("--excerpt-seconds", type=float, default=15.0)
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def load_tracks(jsonl_path: Path):
    """Reassemble per-track token streams from ordered chunks."""
    by_track = defaultdict(list)
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            by_track[rec["metadata"]["track_id"]].append(rec)
    tracks = []
    for tid, recs in by_track.items():
        recs.sort(key=lambda r: r["metadata"].get("chunk_idx", 0))
        meta = recs[0]["metadata"]
        tokens = [t for r in recs for t in r["seq"]]
        tracks.append({"track_id": tid, "artist": meta["artist"],
                       "album": meta.get("album", ""), "title": meta.get("title", ""),
                       "tokens": tokens, "n_chunks": len(recs)})
    tracks.sort(key=lambda t: (t["artist"], t["title"]))
    return tracks


def prepare_ids(tokens, tokenizer):
    special = {tokenizer.eos_tok, tokenizer.bos_tok, tokenizer.pad_tok}
    clean = [tuple(t) if isinstance(t, list) else t for t in tokens]
    return tokenizer.encode([t for t in clean if t not in special])


def sliding_windows(n_ids: int, stride: int):
    body = WINDOW_SIZE - 1  # one slot reserved for EOS
    if n_ids <= body:
        return [(0, n_ids)]
    windows = []
    i = 0
    while i + body <= n_ids:
        windows.append((i, i + body))
        i += stride
    if windows[-1][1] != n_ids:
        windows.append((n_ids - body, n_ids))
    return windows


@torch.no_grad()
def classify_windows(model, ids, windows, eos_id, pad_id, device, batch_size):
    logits_out, seqs, positions = [], [], []

    def flush():
        if not seqs:
            return
        out = model(torch.tensor(seqs, dtype=torch.long, device=device))
        for b, pos in enumerate(positions):
            logits_out.append(out[b, pos].float().cpu().numpy())
        seqs.clear()
        positions.clear()

    for s, e in windows:
        seq = list(ids[s:e]) + [eos_id]
        positions.append(len(seq) - 1)
        if len(seq) < WINDOW_SIZE:
            seq = seq + [pad_id] * (WINDOW_SIZE - len(seq))
        seqs.append(seq)
        if len(seqs) >= batch_size:
            flush()
    flush()
    return np.asarray(logits_out)


def compute_scores(logits: np.ndarray, true_class: int, smooth_sigma: float):
    others = np.concatenate([logits[:, :true_class], logits[:, true_class + 1:]], axis=1)
    margin = logits[:, true_class] - others.max(axis=1)
    sigma = margin.std()
    z = (margin - margin.mean()) / sigma if sigma > 1e-8 else np.zeros_like(margin)
    return {"margin": margin, "zscore": z,
            "smoothed": gaussian_filter1d(z, sigma=smooth_sigma)}


def save_excerpt(tokenizer, ids, start, end, out_path: Path, seconds: float):
    """Write the window as MIDI, keeping only its first `seconds` of music."""
    import pretty_midi

    toks = [tokenizer.vocab[i] for i in ids[start:end] if i < len(tokenizer.vocab)]
    prefix = ("prefix", "instrument", "piano")
    if not toks or toks[0] != prefix:
        toks = [prefix] + toks
    tokenizer.detokenize(toks).to_midi().save(str(out_path))
    if not seconds:
        return
    pm = pretty_midi.PrettyMIDI(str(out_path))
    if pm.get_end_time() <= seconds:
        return
    for inst in pm.instruments:
        inst.notes = [n for n in inst.notes if n.start < seconds]
        for n in inst.notes:
            n.end = min(n.end, seconds)
    pm.write(str(out_path))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    artist_to_id = json.loads(args.artist_map.read_text())
    tokenizer = AbsTokenizer()
    eos_id = tokenizer.vocab.index(tokenizer.eos_tok)
    pad_id = tokenizer.vocab.index(tokenizer.pad_tok)

    tracks = load_tracks(args.test_jsonl)
    if args.artist:
        tracks = [t for t in tracks if t["artist"] == args.artist]
    if args.track_contains:
        tracks = [t for t in tracks if args.track_contains.lower() in t["title"].lower()]
    if args.max_tracks:
        tracks = tracks[:args.max_tracks]
    logger.info(f"{len(tracks)} tracks on {device}")

    model, _ = load_classifier(str(args.classifier), model_name=args.model_name,
                               num_classes=len(artist_to_id), device=device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = args.out_dir / "curves"
    curves_dir.mkdir(exist_ok=True)

    summary = []
    for idx, track in enumerate(tracks, 1):
        if track["artist"] not in artist_to_id:
            continue
        ids = prepare_ids(track["tokens"], tokenizer)
        windows = sliding_windows(len(ids), args.stride)
        logits = classify_windows(model, ids, windows, eos_id, pad_id, device, args.batch_size)
        scores = compute_scores(logits, artist_to_id[track["artist"]], args.smooth_sigma)
        positions = np.array([w[0] for w in windows])
        peak_i = int(np.argmax(scores["smoothed"]))
        trough_i = int(np.argmin(scores["smoothed"]))

        slug = f"{slugify(track['artist'])}__{slugify(track['title'])}"
        np.savez_compressed(curves_dir / f"{slug}.npz", positions=positions,
                            margin=scores["margin"].astype(np.float32),
                            zscore=scores["zscore"].astype(np.float32),
                            smoothed=scores["smoothed"].astype(np.float32))

        entry = {
            "track_id": track["track_id"], "artist": track["artist"],
            "title": track["title"], "album": track["album"],
            "n_windows": len(windows), "n_tokens": len(ids),
            "peak_position": int(positions[peak_i]),
            "peak_smoothed_z": float(scores["smoothed"][peak_i]),
            "trough_position": int(positions[trough_i]),
            "trough_smoothed_z": float(scores["smoothed"][trough_i]),
            "margin_mean": float(scores["margin"].mean()),
            "margin_std": float(scores["margin"].std()),
            "curve_file": f"curves/{slug}.npz",
        }

        if args.save_excerpts:
            ex_dir = args.out_dir / "excerpts"
            ex_dir.mkdir(exist_ok=True)
            for label, wi in (("peak", peak_i), ("trough", trough_i)):
                s, e = windows[wi]
                path = ex_dir / f"{slug}__{label}.mid"
                try:
                    save_excerpt(tokenizer, ids, s, e, path, args.excerpt_seconds)
                    entry[f"{label}_excerpt"] = f"excerpts/{path.name}"
                except Exception as exc:
                    logger.warning(f"{slug} {label} excerpt failed: {exc}")

        summary.append(entry)
        logger.info(f"[{idx}/{len(tracks)}] {track['artist']} - {track['title'][:40]}: "
                    f"peak z={entry['peak_smoothed_z']:+.2f} @ {entry['peak_position']}, "
                    f"trough z={entry['trough_smoothed_z']:+.2f}")

    peaks = [e["peak_smoothed_z"] for e in summary]
    stats = {"n_tracks": len(summary),
             "median_peak_z": float(np.median(peaks)) if peaks else 0.0,
             "max_peak_z": float(np.max(peaks)) if peaks else 0.0,
             "tracks_above_1.5": int(sum(p > 1.5 for p in peaks))}
    (args.out_dir / "summary.json").write_text(
        json.dumps({"stats": stats, "config": {"stride": args.stride,
                                               "smooth_sigma": args.smooth_sigma,
                                               "window": WINDOW_SIZE},
                    "tracks": summary}, indent=1))
    logger.info(f"median peak z={stats['median_peak_z']:.2f}, "
                f"max={stats['max_peak_z']:.2f}, "
                f"{stats['tracks_above_1.5']} tracks above 1.5")
    logger.info(f"wrote {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
