#!/usr/bin/env python3
"""Build MIDI + clip indexes for the from-scratch ResNet transfer experiment.

Arms (Cheston ResNet-50 architecture + recipe, OUR data and splits):
  A (real):  PiJAMA-12 real train/val tracks, reconstructed from the 4096-token
             chunk JSONLs (concat chunks per track_id in chunk_idx order).
  B (synth): clean-room synthetic generations (chunk 0 dropped, generation-level
             val split), reconstructed per generation from the clean-room JSONLs.
  test:      real PiJAMA-12 test tracks (shared by both arms), from test.jsonl.

Output: scratch/camera_ready/resnet_transfer/
  midi/<split>/<n>.mid  +  <split>_index.csv (file, label, group, duration_s)
Clips are cut at dataset time (non-overlapping 30 s starts from duration).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from ariautils.tokenizer import AbsTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "scratch" / "camera_ready" / "resnet_transfer"

SOURCES = {
    # split name -> (jsonl path, group key fn, sort key fn)
    "real_train": (REPO_ROOT / ".cache/paper/pijama12_train_4096.jsonl", "track_id", "chunk_idx"),
    "real_val": (REPO_ROOT / ".cache/paper/pijama12_val_4096.jsonl", "track_id", "chunk_idx"),
    "real_test": (REPO_ROOT / "scratch/characteristic_regions/test.jsonl", "track_id", "chunk_idx"),
    "synth_train": (REPO_ROOT / "scratch/camera_ready/cleanroom_data/train.jsonl", "generation_idx", "chunk_idx"),
    "synth_val": (REPO_ROOT / "scratch/camera_ready/cleanroom_data/val.jsonl", "generation_idx", "chunk_idx"),
}
PREFIX = ("prefix", "instrument", "piano")


def to_tuples(seq):
    return [tuple(t) if isinstance(t, list) else t for t in seq]


def main():
    tokenizer = AbsTokenizer()
    for split, (path, group_key, sort_key) in SOURCES.items():
        groups = defaultdict(list)
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                m = rec["metadata"]
                groups[m[group_key]].append((m.get(sort_key, 0), m["artist"].replace("_", " "), rec["seq"]))
        mididir = OUT / "midi" / split
        mididir.mkdir(parents=True, exist_ok=True)
        rows, skipped = [], 0
        for i, (gid, chunks) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
            chunks.sort(key=lambda c: c[0])
            artist = chunks[0][1]
            tokens = []
            for _idx, _artist, seq in chunks:
                tokens.extend(to_tuples(seq))
            if not tokens or tokens[0] != PREFIX:
                tokens = [PREFIX] + tokens
            try:
                midi = tokenizer.detokenize(tokens).to_midi()
                dst = mididir / f"{i:05d}.mid"
                midi.save(str(dst))
                # duration from pretty_midi for a consistent reading
                from pretty_midi import PrettyMIDI
                dur = PrettyMIDI(str(dst)).get_end_time()
                rows.append({"file": str(dst.relative_to(OUT)), "label": artist,
                             "group": str(gid), "duration_s": round(dur, 2)})
            except Exception as e:
                skipped += 1
        with open(OUT / f"{split}_index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "label", "group", "duration_s"])
            w.writeheader()
            w.writerows(rows)
        total_dur = sum(r["duration_s"] for r in rows) / 3600
        print(f"{split}: {len(rows)} files ({skipped} skipped), {total_dur:.1f} h")


if __name__ == "__main__":
    main()
