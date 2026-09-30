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
# 4096-token chunks: conditional generation, perplexity, and the agreement protocol
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

`--midi-root` is the PiJAMA checkout itself: the CSV's `midi_filepath` values
already begin with `data/midi/`.

Each writes `train.jsonl`, `val.jsonl`, `test.jsonl` and `artist_to_id.json`.
One JSONL record per chunk:

```json
{"seq": [["prefix", "instrument", "piano"], ["piano", 44, 60], ["onset", 0], ...],
 "metadata": {"artist": "Art Tatum", "album": "...", "title": "...",
              "track_id": "...", "midi_filepath": "...", "chunk_idx": 0,
              "split": "test"}}
```

Expected sizes:

| Variant | train | val | test |
|---|---|---|---|
| 4096 tokens | 1,354 | 166 | 177 |
| 1024 tokens | 4,427 | 551 | 591 |

Splits are at the song level (the `song_split` column; 623 / 76 / 80 songs),
so no performance appears in more than one split. Recordings are trimmed to
the CSV's `performance_start_sec` / `performance_end_sec` by default, which
drops applause and spoken introductions on live recordings; `--no-trim`
keeps whole files.

## Which recordings

`pijama12.csv` holds the 779 solo performances by the twelve pianists. PiJAMA
also contains a small number of recordings with ensemble playing (a bassist or
drummer joining on a few tracks); the 19 such recordings by these twelve
pianists are excluded.

## Clip data for the ResNet-50 transfer experiment

`scripts/analysis/build_resnet_transfer_clips.py` converts token JSONL back
into MIDI files, index CSVs, and the artist map the ResNet trainer reads; the
trainer cuts 30-second clips at load time. Point it at the output with
`RESNET_DATA_DIR`.
