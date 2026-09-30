#!/usr/bin/env python3
"""Top-2 piano-roll Jaccard: is any generation locked onto ONE training track?

Memorization signature (cf. the top-2 distance ratio of arXiv 2511.07268 and
the originality-report control design of Yin et al. 2022): a memorized output
is anomalously close to a single specific training piece, while stylistic
similarity is diffuse. For each query sequence we compute the max windowed
piano-roll Jaccard against EACH same-artist training TRACK separately, then
report top-1, top-2, their ratio, and the top-1 track identity. Calibration:
the same computation for real held-out val performances (the natural null).

Window: 8000 ms (the paper-cited scale), 50 ms bins, stride = half window.
Inputs are local JSONL files. Output: results/jaccard_top2/{name}.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llama_pijama.analysis.memorization import extract_notes_from_tuples

REPO_ROOT = Path(__file__).resolve().parents[2]
RES_MS = 50
WINDOW_MS = 8000
W_BINS = WINDOW_MS // RES_MS
STEP = W_BINS // 2


def notes_to_roll(notes) -> np.ndarray:
    if not notes:
        return np.zeros((128, 0), dtype=np.uint8)
    end_ms = max(n.onset_ms + n.duration_ms for n in notes)
    roll = np.zeros((128, end_ms // RES_MS + 1), dtype=np.uint8)
    for n in notes:
        lo = n.onset_ms // RES_MS
        hi = max(lo + 1, (n.onset_ms + n.duration_ms) // RES_MS)
        if 0 <= n.pitch < 128:
            roll[n.pitch, lo:hi] = 1
    return roll


def windows_of(roll: np.ndarray) -> np.ndarray:
    if roll.shape[1] < W_BINS:
        pad = np.zeros((roll.shape[0], W_BINS - roll.shape[1]), dtype=roll.dtype)
        roll = np.concatenate([roll, pad], axis=1)
    starts = range(0, roll.shape[1] - W_BINS + 1, STEP)
    w = np.stack([roll[:, s:s + W_BINS].reshape(-1) for s in starts])
    active = w.sum(axis=1) > 0
    return w[active] if active.any() else w[:1]


def max_jaccard(qw: np.ndarray, rw: np.ndarray) -> float:
    best = 0.0
    for q in qw:
        inter = (q & rw).sum(axis=1)
        union = (q | rw).sum(axis=1)
        valid = union > 0
        if valid.any():
            best = max(best, float((inter[valid] / union[valid]).max()))
    return best


def load_grouped(jsonl: Path, artist_key="artist", group_key="track_id"):
    """-> {artist: {group: [window-matrices]}}"""
    out = defaultdict(lambda: defaultdict(list))
    with open(jsonl) as f:
        for line in f:
            rec = json.loads(line)
            m = rec["metadata"]
            artist = m[artist_key].replace("_", " ")
            notes = extract_notes_from_tuples(rec["seq"])
            if not notes:
                continue
            out[artist][str(m[group_key])].append(windows_of(notes_to_roll(notes)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", required=True, help="jsonl of query sequences (seq + metadata)")
    ap.add_argument("--query-group", default=None,
                    help="metadata key grouping query chunks (e.g. track_id); default: each record separate")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref-jsonl", required=True,
                    help="reference set to scan against (default: train)")
    ap.add_argument("--out-dir", default="results/jaccard_top2",
                    help="directory for the result JSON")
    ap.add_argument("--source-map", default=None,
                    help="JSON mapping query id -> the track_id it was prompted from "
                         "(e.g. the generation provenance). Adds top1 excluding each "
                         "query's own source recording: the paper's 0.20")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"loading reference tracks from {args.ref_jsonl} ...")
    train = load_grouped(Path(args.ref_jsonl))
    for a in train:
        print(f"  {a}: {len(train[a])} tracks")

    print("loading queries...")
    queries = []  # (qid, artist, [window matrices])
    if args.query_group:
        grouped = load_grouped(Path(args.query), group_key=args.query_group)
        for artist, groups in grouped.items():
            for gid, mats in groups.items():
                queries.append((gid, artist, mats))
    else:
        with open(Path(args.query)) as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                m = rec["metadata"]
                notes = extract_notes_from_tuples(rec["seq"])
                if notes:
                    queries.append((str(m.get("generation_idx", i)),
                                    m["artist"].replace("_", " "),
                                    [windows_of(notes_to_roll(notes))]))
    if args.limit:
        queries = queries[: args.limit]
    print(f"{len(queries)} queries")

    rows = []
    for qi, (qid, artist, qmats) in enumerate(queries):
        per_track = {}
        for tid, tmats in train.get(artist, {}).items():
            best = 0.0
            for qm in qmats:
                for tm in tmats:
                    best = max(best, max_jaccard(qm, tm))
            per_track[tid] = best
        if len(per_track) < 2:
            continue
        ranked = sorted(per_track.items(), key=lambda kv: -kv[1])
        (t1, v1), (_t2, v2) = ranked[0], ranked[1]
        rows.append({"query": qid, "artist": artist, "top1": round(v1, 4),
                     "top2": round(v2, 4), "ratio": round(v1 / max(v2, 1e-9), 3),
                     "top1_track": t1})
        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{len(queries)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = np.array([r["ratio"] for r in rows])
    tops = np.array([r["top1"] for r in rows])
    summary = {
        "n": len(rows), "window_ms": WINDOW_MS,
        "top1": {"mean": round(float(tops.mean()), 4), "max": round(float(tops.max()), 4),
                 "p95": round(float(np.percentile(tops, 95)), 4)},
        "ratio": {"mean": round(float(ratios.mean()), 3), "median": round(float(np.median(ratios)), 3),
                  "p95": round(float(np.percentile(ratios, 95)), 3), "max": round(float(ratios.max()), 3)},
        "n_ratio_gt_1.5": int((ratios > 1.5).sum()), "n_ratio_gt_2": int((ratios > 2.0).sum()),
    }
    if args.source_map:
        # A continuation naturally resembles the recording it was prompted from;
        # the memorization question is closeness to any OTHER training track.
        source = {str(k): (v["track_id"] if isinstance(v, dict) else v)
                  for k, v in json.loads(Path(args.source_map).read_text()).items()}
        excl = np.array([r["top2"] if r["top1_track"] == source.get(r["query"]) else r["top1"]
                         for r in rows])
        summary["top1_excl_own_source"] = {
            "mean": round(float(excl.mean()), 4), "max": round(float(excl.max()), 4),
            "p95": round(float(np.percentile(excl, 95)), 4)}
    (out_dir / f"{args.name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
