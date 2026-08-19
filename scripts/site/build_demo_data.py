#!/usr/bin/env python3
"""Pack a set of MIDI continuations into one compact JSON for the demo site.

All inputs are expected to be continuations of the *same* prompt, so the
shared head is detected automatically and stored once. Note data is quantized
to milliseconds and emitted as flat integer arrays, which keeps the payload
small enough to ship with the page.

Example:
    python scripts/site/build_demo_data.py \
        --midi-dir supplementary_materials/aint_misbehavin_continuations \
        --out docs/data/aint_misbehavin.json --title "Ain't Misbehavin'"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pretty_midi


def notes_of(path: Path):
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def shared_prefix_length(tracks) -> int:
    """Number of leading notes identical across every track."""
    if not tracks:
        return 0
    shortest = min(len(t) for t in tracks)
    ref = tracks[0]
    for i in range(shortest):
        key = (round(ref[i].start, 3), ref[i].pitch)
        if any((round(t[i].start, 3), t[i].pitch) != key for t in tracks[1:]):
            return i
    return shortest


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--midi-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--prompt-label", default="prompt")
    return ap.parse_args()


def main():
    args = parse_args()
    paths = sorted(args.midi_dir.glob("*.mid"))
    if not paths:
        raise SystemExit(f"no .mid files in {args.midi_dir}")

    tracks = [notes_of(p) for p in paths]
    n_shared = shared_prefix_length(tracks)
    branch_s = tracks[0][n_shared - 1].end if n_shared else 0.0

    def pack(notes):
        # flat [start_ms, dur_ms, pitch, velocity] * n
        out = []
        for n in notes:
            out += [int(round(n.start * 1000)), int(round((n.end - n.start) * 1000)),
                    int(n.pitch), int(n.velocity)]
        return out

    artists = []
    for path, notes in zip(paths, tracks):
        artists.append({
            "name": path.stem.replace("_", " "),
            "slug": path.stem,
            "notes": pack(notes),
            "duration_ms": int(round(max(n.end for n in notes) * 1000)),
            "n_notes": len(notes),
        })

    payload = {
        "title": args.title or args.midi_dir.name,
        "prompt_label": args.prompt_label,
        "shared_notes": n_shared,
        "branch_ms": int(round(branch_s * 1000)),
        "note_format": ["start_ms", "duration_ms", "pitch", "velocity"],
        "artists": artists,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = args.out.stat().st_size / 1024
    print(f"{len(artists)} continuations, shared prompt = {n_shared} notes "
          f"({branch_s:.2f}s), {size_kb:.0f} KB -> {args.out}")


if __name__ == "__main__":
    main()
