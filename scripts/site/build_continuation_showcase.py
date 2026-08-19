#!/usr/bin/env python3
"""Score existing generated continuations for the companion site.

Takes the generations produced by the conditional model (the same synthetic
corpus the paper uses) and scores each the way the paper scores style
persistence: a 1024-token classification window slides along the
continuation at stride 128, and we record which pianist the classifier hears
in each window.

Several candidates per pianist are scored; the two highest-scoring distinct
samples are kept, each carrying the score the classifier gave it.

Output is a single JSON holding, per pianist: the generated notes, the
per-window classifier verdicts, the overall agreement, and the source track
the continuation was prompted from. The site uses it for playback, for the
score display, and for the blindfold test.

Example:
    python scripts/site/build_continuation_showcase.py \
        --generations .cache/paper/synthetic_tokens/generated_tokens.jsonl \
        --provenance .cache/paper/synthetic_tokens/generation_provenance.json \
        --classifier checkpoints/pijama12_classifier.pt \
        --artist-map data/eval/artist_to_id.json \
        --out docs/data/continuations.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import jsonlines
import torch
from ariautils.tokenizer import AbsTokenizer

from llama_pijama.evaluation import load_model as load_classifier
from llama_pijama.utils.generation import ids_to_tokens, tokens_to_ids

logger = logging.getLogger("showcase")

CLASSIFY_WINDOW = 1024
WINDOW_STRIDE = 128
PREFIX_TOK = ("prefix", "instrument", "piano")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--generations", type=Path, required=True,
                    help="JSONL of generated_ids + artist")
    ap.add_argument("--provenance", type=Path, default=None,
                    help="JSON mapping sample index -> source track")
    ap.add_argument("--train-jsonl", type=Path, default=None,
                    help="the sequences the generations were prompted from; "
                         "when given, each take includes its actual prompt")
    ap.add_argument("--prompt-length", type=int, default=256,
                    help="prompt tokens used at generation time")
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--artist-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score-cache", type=Path, default=None,
                    help="reuse (or write) per-candidate scores, so changing "
                         "--select does not re-run the classifier")
    ap.add_argument("--candidates-per-artist", type=int, default=8,
                    help="how many generations to score per pianist")
    ap.add_argument("--takes-per-artist", type=int, default=2,
                    help="how many of the top-scoring candidates to keep")
    ap.add_argument("--min-windows", type=int, default=5,
                    help="a take must span at least this many classification "
                         "windows (rules out early-EOS stubs)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--artists", nargs="*", default=None, help="default: all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def notes_from_ids(ids, tokenizer, offset_ms=0):
    """Decode token ids to [start_ms, dur_ms, pitch, velocity] quadruples."""
    toks = ids_to_tokens(list(ids), tokenizer)
    if not toks or toks[0] != PREFIX_TOK:
        toks = [PREFIX_TOK] + toks
    try:
        midi = tokenizer.detokenize(toks).to_midi()
    except Exception as exc:
        logger.warning(f"detokenize failed: {exc}")
        return [], 0
    import pretty_midi, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        midi.save(tmp.name)
        path = tmp.name
    try:
        pm = pretty_midi.PrettyMIDI(path)
    finally:
        os.unlink(path)
    flat, end = [], 0.0
    for inst in pm.instruments:
        for n in sorted(inst.notes, key=lambda x: (x.start, x.pitch)):
            flat += [int(round(n.start * 1000)) + offset_ms,
                     int(round((n.end - n.start) * 1000)), int(n.pitch), int(n.velocity)]
            end = max(end, n.end)
    return flat, int(round(end * 1000)) + offset_ms


@torch.no_grad()
def score_windows(classifier, ids, eos_id, device, id_to_artist, true_id):
    """Slide the classifier along a continuation; report what it hears where."""
    windows = []
    n = (len(ids) - CLASSIFY_WINDOW) // WINDOW_STRIDE + 1
    for w in range(max(n, 0)):
        start = w * WINDOW_STRIDE
        seq = list(ids[start:start + CLASSIFY_WINDOW - 1]) + [eos_id]
        eos_pos = len(seq) - 1
        padded = seq + [0] * (CLASSIFY_WINDOW - len(seq))
        logits = classifier(torch.tensor([padded], dtype=torch.long, device=device))[0, eos_pos]
        probs = torch.softmax(logits.float(), dim=-1)
        pred = int(probs.argmax())
        windows.append({
            "position": start,
            "pred": id_to_artist[pred],
            "correct": pred == true_id,
            "confidence": round(float(probs[pred]), 3),
        })
    agreement = (sum(w["correct"] for w in windows) / len(windows)) if windows else 0.0
    return windows, round(agreement, 4)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    artist_to_id = json.loads(args.artist_map.read_text())
    id_to_artist = {v: k for k, v in artist_to_id.items()}
    tokenizer = AbsTokenizer()
    eos_id = tokenizer.vocab.index(tokenizer.eos_tok)
    provenance = json.loads(args.provenance.read_text()) if args.provenance else {}

    # Recover each sample's exact prompt. The generation job walked the train
    # set in order, skipping some records; provenance keeps that order, so an
    # order-preserving match on track_id recovers the source record uniquely.
    prompt_of = {}
    if args.train_jsonl and provenance:
        train = []
        with jsonlines.open(args.train_jsonl) as reader:
            for rec in reader:
                m = rec["metadata"]
                train.append((m.get("track_id") or m.get("midi_filepath"), rec["seq"]))
        ti = 0
        for k in sorted(provenance, key=int):
            want = provenance[k]["track_id"]
            j = ti
            while j < len(train) and train[j][0] != want:
                j += 1
            if j == len(train):
                logger.warning(f"no prompt source found for sample {k}")
                continue
            prompt_of[int(k)] = train[j][1]
            ti = j + 1
        logger.info(f"recovered prompt sources for {len(prompt_of)}/{len(provenance)} samples")

    by_artist = defaultdict(list)
    with jsonlines.open(args.generations) as reader:
        for i, rec in enumerate(reader):
            artist = rec.get("artist")
            if artist in artist_to_id:
                by_artist[artist].append((i, rec))
    artists = args.artists or sorted(by_artist)
    logger.info(f"{sum(len(v) for v in by_artist.values())} generations on file; "
                f"{len(artists)} pianists; scoring on {device}")

    cache = {}
    if args.score_cache and args.score_cache.exists():
        cache = json.loads(args.score_cache.read_text())
        logger.info(f"reusing scores for {len(cache)} pianists from {args.score_cache}")

    classifier = None
    if any(a not in cache for a in artists):
        classifier, _ = load_classifier(str(args.classifier), model_name=args.model_name,
                                        num_classes=len(artist_to_id), device=device)

    items = []
    for artist in artists:
        pool = by_artist[artist]
        random.shuffle(pool)
        by_index = {i: rec for i, rec in pool}
        candidates = []
        if artist in cache:
            for entry in cache[artist]:
                rec = by_index.get(entry["index"])
                if rec is None:
                    continue
                candidates.append((entry["agreement"], entry["index"], rec,
                                   rec["generated_ids"][:args.max_tokens], entry["windows"]))
        else:
            for idx, rec in pool[:args.candidates_per_artist]:
                ids = rec["generated_ids"][:args.max_tokens]
                windows, agreement = score_windows(classifier, ids, eos_id, device,
                                                   id_to_artist, artist_to_id[artist])
                candidates.append((agreement, idx, rec, ids, windows))
                logger.info(f"  {artist} #{idx}: {agreement:.1%} over {len(windows)} windows")
            cache[artist] = [{"index": c[1], "agreement": c[0], "windows": c[4]}
                             for c in candidates]
        if not candidates:
            continue
        # Early-EOS stubs are not demo material: a take must be long enough
        # to carry a meaningful score. Ties on agreement break toward the
        # take with more windows behind it.
        eligible = [c for c in candidates if len(c[4]) >= args.min_windows]
        dropped = len(candidates) - len(eligible)
        if dropped:
            logger.info(f"  {artist}: dropped {dropped} short candidate(s)")
        eligible.sort(key=lambda c: (-c[0], -len(c[4])))
        keep = eligible[:args.takes_per_artist]

        takes = []
        for agreement, idx, rec, ids, windows in keep:
            sample_idx = rec.get("sample_idx", idx)
            src = provenance.get(str(sample_idx), {})
            prompt_notes, branch = [], 0
            seq = prompt_of.get(sample_idx)
            if seq is not None:
                prompt_ids = tokens_to_ids(seq, tokenizer)[:args.prompt_length]
                # decode prompt and prompt+continuation as one stream so the
                # join keeps continuous timing, then split at the branch
                _, branch = notes_from_ids(prompt_ids, tokenizer)
                combined, _ = notes_from_ids(prompt_ids + list(ids), tokenizer)
                prompt_notes, notes = [], []
                for i in range(0, len(combined), 4):
                    (prompt_notes if combined[i] < branch else notes).extend(combined[i:i+4])
                duration = branch
                if notes:
                    duration = max(notes[i] + notes[i+1] for i in range(0, len(notes), 4))
            else:
                notes, duration = notes_from_ids(ids, tokenizer)
            takes.append({
                "sample_idx": sample_idx,
                "prompt_title": Path(src.get("track_id", "")).stem,
                "prompt_notes": prompt_notes, "branch_ms": branch,
                "notes": notes, "duration_ms": duration,
                "agreement": agreement, "windows": windows,
                "generated_tokens": len(ids),
            })
        items.append({"artist": artist, "takes": takes})
        logger.info(f"{artist}: kept {[f'{t[0]:.0%}' for t in keep]} "
                    f"(candidates {sorted(round(c[0], 2) for c in candidates)})")

    if args.score_cache:
        args.score_cache.parent.mkdir(parents=True, exist_ok=True)
        args.score_cache.write_text(json.dumps(cache))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {"source": str(args.generations),
                   "classify_window": CLASSIFY_WINDOW, "window_stride": WINDOW_STRIDE,
                   "candidates_per_artist": args.candidates_per_artist,
                   "selection": "top scoring distinct samples", "seed": args.seed},
        "note_format": ["start_ms", "duration_ms", "pitch", "velocity"],
        "items": items,
    }, separators=(",", ":")))
    logger.info(f"wrote {len(items)} continuations, "
                f"{args.out.stat().st_size/1024:.0f} KB -> {args.out}")


if __name__ == "__main__":
    main()
