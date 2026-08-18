"""Tokenization helpers shared by the dataset builders.

Chunking follows Aria's pretraining data format: middle chunks are full
`max_seq_len` windows with no EOS; only the final chunk of a performance
carries one.
"""
from __future__ import annotations

import mido
from ariautils.tokenizer import AbsTokenizer


def trim_midi_to_time_range(
    midi_path: str,
    start_sec: float,
    end_sec: float,
) -> mido.MidiFile:
    """Trim a MIDI file to a specific time range in seconds.

    Args:
        midi_path: Path to the input MIDI file
        start_sec: Start time in seconds (inclusive)
        end_sec: End time in seconds (exclusive)

    Returns:
        A new mido.MidiFile containing only events within the time range,
        with timing adjusted so the trimmed segment starts at time 0.
    """
    mid = mido.MidiFile(midi_path)
    trimmed_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        trimmed_mid.tracks.append(new_track)

        # First pass: convert to absolute time and track tempo changes
        abs_time_ticks = 0
        current_tempo = 500000  # Default 120 BPM
        events_with_time = []

        for msg in track:
            abs_time_ticks += msg.time
            events_with_time.append({
                "abs_ticks": abs_time_ticks,
                "msg": msg.copy(),
                "tempo_at_event": current_tempo,
            })
            if msg.type == "set_tempo":
                current_tempo = msg.tempo

        # Second pass: convert ticks to seconds and filter
        # We need cumulative time accounting for tempo changes
        filtered_events = []
        prev_ticks = 0
        prev_tempo = 500000
        cumulative_sec = 0.0

        # Pre-event meta messages to keep (tempo, time signature at start)
        pre_start_meta = []

        for evt in events_with_time:
            # Calculate time delta in seconds
            delta_ticks = evt["abs_ticks"] - prev_ticks
            if delta_ticks > 0:
                delta_sec = mido.tick2second(delta_ticks, mid.ticks_per_beat, prev_tempo)
                cumulative_sec += delta_sec

            msg = evt["msg"]

            # Update tempo for next iteration
            if msg.type == "set_tempo":
                prev_tempo = msg.tempo

            prev_ticks = evt["abs_ticks"]

            # Keep important meta messages from before the start
            if cumulative_sec < start_sec and msg.is_meta and msg.type in [
                "set_tempo", "time_signature", "key_signature", "track_name"
            ]:
                # Only keep the latest of each type
                pre_start_meta = [m for m in pre_start_meta if m["msg"].type != msg.type]
                pre_start_meta.append({"sec": 0, "msg": msg.copy()})

            # Filter events within the time range
            if start_sec <= cumulative_sec < end_sec:
                filtered_events.append({
                    "sec": cumulative_sec,
                    "msg": msg,
                })

        # Combine pre-start meta with filtered events
        all_events = pre_start_meta + filtered_events

        if not all_events:
            continue

        # Find the minimum time for events in range (offset for time=0)
        min_sec = start_sec

        # Convert back to delta times
        current_tempo_for_conversion = 500000
        prev_sec_adjusted = 0.0

        for evt in all_events:
            msg = evt["msg"]
            adjusted_sec = max(0, evt["sec"] - min_sec) if evt["sec"] >= start_sec else 0

            delta_sec = adjusted_sec - prev_sec_adjusted
            if delta_sec < 0:
                delta_sec = 0

            delta_ticks = mido.second2tick(delta_sec, mid.ticks_per_beat, current_tempo_for_conversion)
            msg.time = int(round(delta_ticks))

            new_track.append(msg)
            prev_sec_adjusted = adjusted_sec

            if msg.type == "set_tempo":
                current_tempo_for_conversion = msg.tempo

    return trimmed_mid


def _ensure_eos_and_truncate(seq: list, tokenizer: AbsTokenizer, max_seq_len: int) -> list:
    seq = seq[:max_seq_len]
    if tokenizer.eos_tok not in seq:
        if len(seq) == 0:
            seq = [tokenizer.eos_tok]
        else:
            seq[-1] = tokenizer.eos_tok
    return seq


def _chunk_sequence(seq: list, tokenizer: AbsTokenizer, max_seq_len: int, stride: int = None) -> list[list]:
    """Chunk a sequence into multiple windows of max_seq_len tokens.

    Only the final chunk of a piece gets an EOS token appended. Middle
    chunks are full max_seq_len tokens with no EOS, matching Aria's
    pretraining data format.

    Args:
        seq: Full token sequence
        tokenizer: Tokenizer for special tokens
        max_seq_len: Maximum sequence length per chunk
        stride: Stride for sliding window. If None, uses non-overlapping chunks (stride=max_seq_len)

    Returns:
        List of chunked sequences. Only the last chunk has EOS.
    """
    if stride is None:
        stride = max_seq_len

    # Remove BOS/EOS if present to handle them properly
    clean_seq = [tok for tok in seq if tok not in (tokenizer.bos_tok, tokenizer.eos_tok)]

    if len(clean_seq) == 0:
        return [[tokenizer.eos_tok]]

    chunks = []
    for i in range(0, len(clean_seq), stride):
        chunk = clean_seq[i:i + max_seq_len]
        if len(chunk) > 0:
            chunks.append(chunk)

    # Only add EOS to the final chunk
    if chunks:
        last = chunks[-1]
        if len(last) >= max_seq_len:
            # Last chunk is full — truncate to make room for EOS
            last = last[:max_seq_len - 1]
        last.append(tokenizer.eos_tok)
        chunks[-1] = last

    return chunks if chunks else [[tokenizer.eos_tok]]
