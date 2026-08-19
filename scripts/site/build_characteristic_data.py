#!/usr/bin/env python3
"""Pack characteristic-region curves and excerpts for the companion site.

Consumes the output directory of scripts/characteristic_regions.py (the
per-track score curves and the peak/trough MIDI excerpts) and produces one
JSON the site can play: for each track, the smoothed z-score curve plus the
notes of its most and least characteristic regions.

Example:
    python scripts/site/build_characteristic_data.py \
        --regions-dir results/characteristic_regions \
        --out docs/data/characteristic.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pretty_midi


def pack_notes(path: Path):
    if not path.exists():
        return [], 0
    pm = pretty_midi.PrettyMIDI(str(path))
    flat, end = [], 0.0
    for inst in pm.instruments:
        for n in sorted(inst.notes, key=lambda x: (x.start, x.pitch)):
            flat += [int(round(n.start * 1000)), int(round((n.end - n.start) * 1000)),
                     int(n.pitch), int(n.velocity)]
            end = max(end, n.end)
    return flat, int(round(end * 1000))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--regions-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-curve-points", type=int, default=400,
                    help="downsample long curves to keep the payload small")
    return ap.parse_args()


def main():
    args = parse_args()
    summary = json.loads((args.regions_dir / "summary.json").read_text())
    tracks = []
    for t in summary["tracks"]:
        curve_path = args.regions_dir / t["curve_file"]
        if not curve_path.exists():
            continue
        z = np.load(curve_path)
        smoothed = z["smoothed"].astype(float)
        positions = z["positions"].astype(int)
        if len(smoothed) > args.max_curve_points:
            idx = np.linspace(0, len(smoothed) - 1, args.max_curve_points).astype(int)
            smoothed, positions = smoothed[idx], positions[idx]

        entry = {
            "artist": t["artist"], "title": t["title"],
            "peak_z": round(t["peak_smoothed_z"], 2),
            "trough_z": round(t["trough_smoothed_z"], 2),
            "peak_position": t["peak_position"], "trough_position": t["trough_position"],
            "n_tokens": t["n_tokens"],
            "positions": positions.tolist(),
            "curve": [round(v, 3) for v in smoothed],
        }
        for label in ("peak", "trough"):
            key = f"{label}_excerpt"
            if key in t:
                notes, dur = pack_notes(args.regions_dir / t[key])
                entry[f"{label}_notes"] = notes
                entry[f"{label}_duration_ms"] = dur
        if "peak_notes" in entry and "trough_notes" in entry:
            tracks.append(entry)

    tracks.sort(key=lambda e: -e["peak_z"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"stats": summary["stats"], "tracks": tracks},
                                   separators=(",", ":")))
    print(f"{len(tracks)} tracks, {args.out.stat().st_size/1024:.0f} KB -> {args.out}")


if __name__ == "__main__":
    main()
