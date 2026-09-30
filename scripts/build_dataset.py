#!/usr/bin/env python3
"""Turn PiJAMA MIDI into the tokenized JSONL splits used for training and evaluation.

Reads a metadata CSV (one row per performance, with `artist`, `midi_filepath`
and a split column), tokenizes each MIDI with Aria's AbsTokenizer, chunks it
into fixed-length windows, and writes one JSONL record per chunk:

    {"seq": [...tokens...], "metadata": {artist, album, title, track_id,
                                         midi_filepath, chunk_idx, split}}

Chunk length selects the dataset variant: 4096 tokens for generation,
1024 for classification (the ISMIR 2026 paper uses both).

Example:
    python scripts/build_dataset.py \
        --midi-root ~/PiJAMA \
        --metadata-csv data/pijama12.csv \
        --out-dir data/pijama12_4096 --max-seq-len 4096
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

from llama_pijama.data import chunk_sequence, trim_midi_to_time_range

logger = logging.getLogger("build_dataset")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--midi-root", type=Path, required=True,
                    help="The PiJAMA checkout; `midi_filepath` values (data/midi/...) are relative to it")
    ap.add_argument("--metadata-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split-column", default="song_split")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--no-trim", dest="trim_to_performance_bounds", action="store_false",
                    help="Keep whole recordings instead of trimming to "
                         "performance_start_sec/performance_end_sec. The paper's data is "
                         "trimmed (drops applause and announcements on live recordings).")
    ap.add_argument("--write-artist-map", action="store_true", default=True,
                    help="Also write artist_to_id.json over the artists present")
    return ap.parse_args()


def build_split(df, split, args, tokenizer, out_path: Path):
    subset = df[df[args.split_column] == split]
    n_records = n_tracks = n_failed = 0
    with out_path.open("w") as out:
        for _, row in subset.iterrows():
            midi_path = args.midi_root / row["midi_filepath"]
            if not midi_path.exists():
                logger.warning(f"missing MIDI: {midi_path}")
                n_failed += 1
                continue
            try:
                if args.trim_to_performance_bounds and "performance_start_sec" in row:
                    start = float(row.get("performance_start_sec", 0) or 0)
                    end = float(row.get("performance_end_sec", float("inf")) or float("inf"))
                    duration = float(row["duration_sec"]) if "duration_sec" in row and pd.notna(row["duration_sec"]) else None
                    if start > 0 or (duration is not None and end < duration - 1):
                        trimmed = trim_midi_to_time_range(str(midi_path), start, end)
                        tmp = out_path.parent / f".{midi_path.stem}.trimmed.mid"
                        trimmed.save(str(tmp))
                        try:
                            midi_dict = MidiDict.from_midi(str(tmp))
                        finally:
                            tmp.unlink(missing_ok=True)
                    else:
                        midi_dict = MidiDict.from_midi(str(midi_path))
                else:
                    midi_dict = MidiDict.from_midi(str(midi_path))

                tokens = tokenizer.tokenize(midi_dict)
                if tokenizer.dim_tok in tokens:
                    tokens.remove(tokenizer.dim_tok)
                chunks = chunk_sequence(tokens, tokenizer, args.max_seq_len,
                                        stride=args.max_seq_len)
            except Exception as exc:
                logger.warning(f"tokenization failed for {midi_path.name}: {exc}")
                n_failed += 1
                continue

            for idx, chunk in enumerate(chunks):
                out.write(json.dumps({
                    "seq": chunk,
                    "metadata": {
                        "artist": row["artist"],
                        "album": row.get("album", ""),
                        "title": row.get("title", ""),
                        "track_id": row.get("track_id", row["midi_filepath"]),
                        "midi_filepath": row["midi_filepath"],
                        "chunk_idx": idx,
                        "split": split,
                    },
                }) + "\n")
                n_records += 1
            n_tracks += 1
    logger.info(f"{split}: {n_tracks} tracks -> {n_records} chunks "
                f"({n_failed} failed) -> {out_path}")
    return n_records


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    df = pd.read_csv(args.metadata_csv)
    if args.split_column not in df.columns:
        raise SystemExit(f"missing split column '{args.split_column}'; "
                         f"available: {list(df.columns)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AbsTokenizer()

    for split in args.splits:
        build_split(df, split, args, tokenizer, args.out_dir / f"{split}.jsonl")

    if args.write_artist_map:
        artists = sorted(df["artist"].unique())
        (args.out_dir / "artist_to_id.json").write_text(
            json.dumps({a: i for i, a in enumerate(artists)}, indent=1))
        logger.info(f"wrote artist_to_id.json ({len(artists)} artists)")


if __name__ == "__main__":
    main()
