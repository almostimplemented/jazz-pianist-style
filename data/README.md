# Data

## What is here

| File | Purpose |
|---|---|
| `pijama12.csv` | The 12 selected pianists: one row per performance, with album, title, duration, performance boundaries, and the `song_split` column used for all experiments. |
| `pijama_30.csv` | All 30 PiJAMA artists; the artist list backing `ConditionalTokenizer`. |

## Getting the audio and MIDI

PiJAMA is a public dataset of solo jazz piano performances with automatically
transcribed MIDI: <https://github.com/almostimplemented/PiJAMA>

Clone it, then point the dataset builder at the checkout — `midi_filepath` in
the CSVs is relative to the repository root.

## Building the tokenized splits

Two variants are used in the paper, differing only in chunk length:

```bash
# 4096-token chunks: conditional generation and the agreement protocol
python scripts/build_dataset.py \
    --midi-root /path/to/PiJAMA \
    --metadata-csv data/pijama12.csv \
    --out-dir data/pijama12_4096 --max-seq-len 4096

# 1024-token chunks: classification and characteristic regions
python scripts/build_dataset.py \
    --midi-root /path/to/PiJAMA \
    --metadata-csv data/pijama12.csv \
    --out-dir data/pijama12_1024 --max-seq-len 1024
```

Each writes `train.jsonl`, `val.jsonl`, `test.jsonl` and `artist_to_id.json`.
One JSONL record per chunk:

```json
{"seq": [["prefix", "instrument", "piano"], ["piano", 44, 60], ["onset", 0], ...],
 "metadata": {"artist": "Art Tatum", "album": "...", "title": "...",
              "track_id": "...", "midi_filepath": "...", "chunk_idx": 0,
              "split": "test"}}
```

Splits are at the song level, so no performance appears in more than one split.
Add `--trim-to-performance-bounds` to cut applause and announcements from live
recordings using the CSV's `performance_start_sec` / `performance_end_sec`.

## Clip data for the ResNet-50 transfer experiment

`scripts/analysis/build_resnet_transfer_clips.py` converts token JSONL back
into MIDI files plus index CSVs of 30-second clips. Point the training script
at the result with `RESNET_DATA_DIR`.
