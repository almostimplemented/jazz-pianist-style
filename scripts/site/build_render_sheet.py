#!/usr/bin/env python3
"""Lay every clip the companion site plays into one MIDI file for rendering.

The site's data files hold the exact notes each player schedules. This writes
them end to end into a single "render sheet" so the whole set can be bounced
through one piano instrument in a DAW in a single pass. A manifest records
where each clip sits on that timeline; split_render.py cuts the bounced audio
back into per-clip files from it.

Two short sync notes bracket the sheet (a single high C near the start and
after the last clip) so the splitter can measure the bounce's actual offset
and check for tempo drift.

Example:
    python scripts/site/build_render_sheet.py --out render/render_sheet.mid
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pretty_midi

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"

GAP_S = 4.0          # silence between clips, so tails never bleed across a cut
LEAD_S = 3.0         # before the first clip
SYNC_PITCH = 108     # C8: short, bright, easy to detect
SYNC_T = 0.5


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def clips():
    """Yield (id, section, meta, flat_notes, duration_ms) for every clip."""
    am = json.loads((DATA / "aint_misbehavin.json").read_text())
    for a in am["artists"]:
        yield (f"am__{slug(a['name'])}", "aint_misbehavin",
               {"artist": a["name"]}, a["notes"], a["duration_ms"])

    sc = json.loads((DATA / "continuations.json").read_text())
    for it in sc["items"]:
        for i, t in enumerate(it["takes"]):
            yield (f"sc__{slug(it['artist'])}__take{i + 1}", "continuations",
                   {"artist": it["artist"], "take": i, "sample_idx": t["sample_idx"]},
                   t["prompt_notes"] + t["notes"], t["duration_ms"])

    rg = json.loads((DATA / "characteristic.json").read_text())
    for t in rg["tracks"]:
        for which in ("peak", "trough"):
            yield (f"rg__{slug(t['artist'])}__{which}", "regions",
                   {"artist": t["artist"], "which": which},
                   t[f"{which}_notes"], t[f"{which}_duration_ms"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "render" / "render_sheet.mid")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="default: next to --out as render_manifest.json")
    args = ap.parse_args()
    manifest_path = args.manifest or args.out.with_name("render_manifest.json")

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=0, name="Render sheet")
    pm.instruments.append(piano)

    def sync(t):
        piano.notes.append(pretty_midi.Note(velocity=100, pitch=SYNC_PITCH,
                                            start=t, end=t + 0.1))

    sync(SYNC_T)
    t = LEAD_S
    entries = []
    for cid, section, meta, notes, dur_ms in clips():
        end = 0.0
        for i in range(0, len(notes), 4):
            s = t + notes[i] / 1000
            e = s + max(0.02, notes[i + 1] / 1000)
            piano.notes.append(pretty_midi.Note(
                velocity=max(1, min(127, int(notes[i + 3]))),
                pitch=int(notes[i + 2]), start=s, end=e))
            end = max(end, e)
        clip_len = max(dur_ms / 1000, end - t)
        pm.lyrics.append(pretty_midi.Lyric(cid, t))  # visible in most DAWs' event lists
        entries.append({"id": cid, "section": section, **meta,
                        "start_s": round(t, 3), "end_s": round(t + clip_len, 3),
                        "duration_ms": dur_ms, "n_notes": len(notes) // 4})
        t += clip_len + GAP_S
    sync_end = t
    sync(sync_end)
    total = sync_end + 2.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(args.out))
    manifest_path.write_text(json.dumps({
        "tempo_bpm": 120, "sync_pitch": SYNC_PITCH,
        "sync_start_s": SYNC_T, "sync_end_s": round(sync_end, 3),
        "total_s": round(total, 3), "gap_s": GAP_S, "clips": entries,
    }, indent=1))
    print(f"{len(entries)} clips, {total/60:.1f} min -> {args.out}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
