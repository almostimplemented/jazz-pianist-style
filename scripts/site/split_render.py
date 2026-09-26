#!/usr/bin/env python3
"""Cut a bounced render sheet back into per-clip audio files for the site.

Takes the WAV bounced from build_render_sheet.py's MIDI and its manifest.
Locates the two sync notes to correct for any offset or tempo drift in the
bounce, applies one global gain (no per-clip normalisation: dynamics are part
of a pianist's style), cuts each clip with a short tail for the last notes to
ring, and encodes with ffmpeg.

Example:
    python scripts/site/split_render.py \\
        --wav render/render_sheet.wav \\
        --manifest render/render_manifest.json \\
        --out-dir docs/audio --format m4a
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

TAIL_S = 2.5
PEAK_DBFS = -1.0


def to_float(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        return x / 32768.0
    if x.dtype == np.int32:
        return x / 2147483648.0
    if x.dtype == np.uint8:
        return (x - 128) / 128.0
    return x.astype(np.float64)


def onset_near(mono: np.ndarray, sr: int, expect_s: float, window_s: float = 1.5):
    """Time of the first energy rise within ±window of the expected sync note."""
    lo = max(0, int((expect_s - window_s) * sr))
    hi = min(len(mono), int((expect_s + window_s) * sr))
    seg = np.abs(mono[lo:hi])
    if seg.size == 0:
        return None
    thresh = max(seg.max() * 0.2, 1e-4)
    idx = np.argmax(seg > thresh)
    if seg[idx] <= thresh:
        return None
    return (lo + idx) / sr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--format", choices=["m4a", "mp3", "wav"], default="m4a")
    ap.add_argument("--bitrate", default="112k")
    ap.add_argument("--no-sync", action="store_true",
                    help="trust the manifest timestamps as they are")
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    sr, data = wavfile.read(str(args.wav))
    audio = to_float(data)
    if audio.ndim == 1:
        audio = audio[:, None]
    mono = audio.mean(axis=1)
    print(f"{args.wav}: {sr} Hz, {audio.shape[1]} ch, {len(mono)/sr/60:.1f} min")

    # Map manifest time -> bounce time via the two sync notes
    offset, scale = 0.0, 1.0
    if not args.no_sync:
        a = onset_near(mono, sr, man["sync_start_s"])
        b = onset_near(mono, sr, man["sync_end_s"])
        if a is None or b is None:
            raise SystemExit("could not find the sync notes; check the bounce starts "
                             "at bar 1, or pass --no-sync")
        scale = (b - a) / (man["sync_end_s"] - man["sync_start_s"])
        offset = a - man["sync_start_s"] * scale
        print(f"sync: start at {a:.3f}s (expected {man['sync_start_s']}), "
              f"end at {b:.3f}s (expected {man['sync_end_s']}); "
              f"offset {offset*1000:+.0f} ms, scale {scale:.5f}")
        if abs(scale - 1) > 0.002:
            raise SystemExit("tempo drift over 0.2%: the bounce was not at 120 BPM")

    gain = 10 ** (PEAK_DBFS / 20) / max(np.abs(audio).max(), 1e-9)
    print(f"global gain {20*np.log10(gain):+.1f} dB")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    with tempfile.TemporaryDirectory() as tmp:
        for c in man["clips"]:
            s = int((c["start_s"] * scale + offset) * sr)
            e = int(((c["end_s"] + TAIL_S) * scale + offset) * sr)
            clip = np.clip(audio[s:e] * gain, -1, 1)
            fade = min(len(clip), int(0.05 * sr))  # tame the cut at the tail
            if fade:
                clip[-fade:] *= np.linspace(1, 0, fade)[:, None]
            out = args.out_dir / f"{c['id']}.{args.format}"
            if args.format == "wav":
                wavfile.write(str(out), sr, (clip * 32767).astype(np.int16))
            else:
                raw = Path(tmp) / "clip.wav"
                wavfile.write(str(raw), sr, (clip * 32767).astype(np.int16))
                codec = ["-c:a", "aac", "-b:a", args.bitrate] if args.format == "m4a" \
                    else ["-c:a", "libmp3lame", "-b:a", args.bitrate]
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                                *codec, str(out)], check=True)
            index[c["id"]] = {"file": out.name, "duration_ms": c["duration_ms"],
                              "audio_ms": int(len(clip) / sr * 1000)}
            print(f"  {out.name}  {len(clip)/sr:6.1f}s")
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=1))
    total = sum(f.stat().st_size for f in args.out_dir.glob(f"*.{args.format}"))
    print(f"{len(index)} clips, {total/1e6:.1f} MB -> {args.out_dir}")


if __name__ == "__main__":
    main()
