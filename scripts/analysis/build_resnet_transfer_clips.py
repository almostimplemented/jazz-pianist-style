#!/usr/bin/env python3
"""Build MIDI + clip indexes for the from-scratch ResNet transfer experiment.

The ResNet-50 arms use the architecture and recipe of Cheston et al.'s Deep
Pianist Identification on our data and splits:
  real:  PiJAMA-12 real train/val tracks, reconstructed from the 4096-token
         JSONLs (chunks concatenated per track in chunk order)
  synth: generated continuations only, reconstructed per generation
  test:  the real PiJAMA-12 test tracks, shared by both arms

For each split this writes midi/<split>/<n>.mid and <split>_index.csv
(file, label, group, duration_s); train_resnet_transfer.py cuts 30 s clips
from these at load time. Point it at --out-dir with RESNET_DATA_DIR.

Example:
    python scripts/analysis/build_resnet_transfer_clips.py \
        --real-train data/pijama12_4096/train.jsonl \
        --real-val data/pijama12_4096/val.jsonl \
        --real-test data/pijama12_1024/test.jsonl \
        --synth-train data/synthetic/train.jsonl --synth-val data/synthetic/val.jsonl \
        --out-dir data/resnet_transfer
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ariautils.tokenizer import AbsTokenizer
from pretty_midi import PrettyMIDI

PREFIX = ("prefix", "instrument", "piano")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-train", type=Path)
    ap.add_argument("--real-val", type=Path)
    ap.add_argument("--real-test", type=Path)
    ap.add_argument("--synth-train", type=Path, help="generated continuations, 1024-token chunks")
    ap.add_argument("--synth-val", type=Path)
    ap.add_argument("--artist-map", type=Path, default=None,
                    help="artist_to_id.json to copy in (default: sorted labels found)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/resnet_transfer"))
    args = ap.parse_args()
    if not any([args.real_train, args.real_val, args.real_test, args.synth_train, args.synth_val]):
        ap.error("give at least one input split")
    return args


def to_tuples(seq):
    return [tuple(t) if isinstance(t, list) else t for t in seq]


def main():
    args = parse_args()
    out = args.out_dir
    # split name -> (jsonl, key grouping chunks into one piece, key ordering them)
    sources = {
        "real_train": (args.real_train, "track_id", "chunk_idx"),
        "real_val": (args.real_val, "track_id", "chunk_idx"),
        "real_test": (args.real_test, "track_id", "chunk_idx"),
        "synth_train": (args.synth_train, "generation_idx", "chunk_idx"),
        "synth_val": (args.synth_val, "generation_idx", "chunk_idx"),
    }
    tokenizer = AbsTokenizer()
    labels = set()
    for split, (path, group_key, sort_key) in sources.items():
        if path is None:
            continue
        groups = defaultdict(list)
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                m = rec["metadata"]
                groups[m[group_key]].append((m.get(sort_key, 0), m["artist"].replace("_", " "), rec["seq"]))
        mididir = out / "midi" / split
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
                dur = PrettyMIDI(str(dst)).get_end_time()
                rows.append({"file": str(dst.relative_to(out)), "label": artist,
                             "group": str(gid), "duration_s": round(dur, 2)})
            except Exception:
                skipped += 1
        with open(out / f"{split}_index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "label", "group", "duration_s"])
            w.writeheader()
            w.writerows(rows)
        labels.update(r["label"] for r in rows)
        total_dur = sum(r["duration_s"] for r in rows) / 3600
        print(f"{split}: {len(rows)} files ({skipped} skipped), {total_dur:.1f} h")
    # The trainer reads the label order from here
    mapping = (json.loads(args.artist_map.read_text()) if args.artist_map
               else {a: i for i, a in enumerate(sorted(labels))})
    (out / "artist_to_id.json").write_text(json.dumps(mapping, indent=1))
    print(f"artist_to_id.json: {len(mapping)} artists")


if __name__ == "__main__":
    main()
