#!/usr/bin/env python3
"""Generate artist-conditioned piano continuations as MIDI.

Samples from the cross-attention conditional generator, optionally continuing
a prompt MIDI file. Conditioning modes mirror the evaluation protocols in the
ISMIR 2026 paper "Learning Jazz Pianist Style with Cross-Attention
Conditioning":

    conditioned   cross-attention with the true artist context (default)
    ablation      cross-attention weights present, context zeroed
    mismatch      conditioned on a different artist than requested

Example:
    python scripts/generate.py \
        --checkpoint-dir checkpoints/generator_best \
        --artist-map data/eval/artist_to_id.json \
        --artist "Art Tatum" --num-samples 2 --out-dir samples/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

from aria.config import load_model_config
from aria.model import ModelConfig
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

from llama_pijama.models import ArtistEmbedding
from llama_pijama.utils.generation import generate_tokens, generate_tokens_kv, ids_to_tokens

logger = logging.getLogger("generate")

PREFIX_TOK = ("prefix", "instrument", "piano")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, required=True,
                    help="Directory with model.safetensors + artist_embeddings.safetensors")
    ap.add_argument("--artist-map", type=Path, required=True)
    ap.add_argument("--artist", action="append", default=None,
                    help="Artist to condition on (repeatable; default: all)")
    ap.add_argument("--mode", choices=["conditioned", "ablation", "mismatch"],
                    default="conditioned")
    ap.add_argument("--mismatch-artist", default=None,
                    help="Artist whose context is used in --mode mismatch")
    ap.add_argument("--prompt-midi", type=Path, default=None,
                    help="Continue this MIDI file instead of starting from scratch")
    ap.add_argument("--prompt-tokens", type=int, default=None,
                    help="Truncate the prompt to its first N tokens (paper uses 128/256/512)")
    ap.add_argument("--num-samples", type=int, default=1, help="Samples per artist")
    ap.add_argument("--max-length", type=int, default=1024, help="New tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("samples"))
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--ca-layers", type=int, nargs="*", default=[8, 9, 10, 11, 12, 13, 14, 15])
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-kv-cache", action="store_true",
                    help="Disable the KV cache (slower; use if the cached path "
                         "is unavailable on your device)")
    return ap.parse_args()


def load_generator(args, num_artists: int, device: str):
    """Returns (model, artist_embeddings, uses_kv_cache)."""
    cfg = load_model_config(args.model_name)
    cfg.setdefault("resid_dropout", 0.0)
    cfg["grad_checkpoint"] = False
    model_config = ModelConfig(**cfg)
    ca_config = {"layers": list(args.ca_layers), "dropout": 0.0}
    state = load_file(args.checkpoint_dir / "model.safetensors")

    uses_kv = not args.no_kv_cache
    if uses_kv:
        try:
            from llama_pijama.models import CrossAttentionInferenceLM
            model = CrossAttentionInferenceLM(model_config=model_config,
                                              cross_attention_config=ca_config)
            model.load_state_dict(state, strict=True)
            model = model.to(device).eval()
            model.setup_cache(batch_size=1, max_seq_len=args.max_length + 512,
                              dtype=torch.float32)
        except Exception as e:
            logger.warning(f"KV-cached inference unavailable ({e}); falling back")
            uses_kv = False
    if not uses_kv:
        from llama_pijama.models import CrossAttentionTransformerLM
        model = CrossAttentionTransformerLM(model_config, {**ca_config, "gate_init": 0.1})
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()

    emb = ArtistEmbedding(num_artists=num_artists, d_model=model_config.d_model,
                          context_length=args.context_length, dropout=0.0)
    emb.load_state_dict(load_file(args.checkpoint_dir / "artist_embeddings.safetensors"),
                        strict=True)
    emb = emb.to(device).eval()
    return model, emb, uses_kv


def load_prompt_ids(path: Path, tokenizer, limit=None):
    midi_dict = MidiDict.from_midi(str(path))
    toks = [t for t in tokenizer.tokenize(midi_dict) if t != tokenizer.eos_tok]
    ids = []
    for t in toks:
        try:
            ids.append(tokenizer.vocab.index(t))
        except ValueError:
            continue
    if limit:
        ids = ids[:limit]
    return ids


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    artist_to_id = json.loads(args.artist_map.read_text())
    tokenizer = AbsTokenizer()
    model, emb, uses_kv = load_generator(args, len(artist_to_id), device)
    logger.info(f"device={device} kv_cache={uses_kv} mode={args.mode}")

    artists = args.artist or sorted(artist_to_id)
    unknown = [a for a in artists if a not in artist_to_id]
    if unknown:
        raise SystemExit(f"unknown artist(s): {unknown}\navailable: {sorted(artist_to_id)}")

    prompt_ids = None
    if args.prompt_midi:
        prompt_ids = load_prompt_ids(args.prompt_midi, tokenizer, args.prompt_tokens)
        logger.info(f"prompt: {len(prompt_ids)} tokens from {args.prompt_midi.name}")

    eos_id = tokenizer.vocab.index(tokenizer.eos_tok)
    prefix_id = tokenizer.vocab.index(PREFIX_TOK)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for artist in artists:
        # Which artist's context actually drives generation
        ctx_artist = artist
        if args.mode == "mismatch":
            others = [a for a in sorted(artist_to_id) if a != artist]
            ctx_artist = args.mismatch_artist or others[0]
            if ctx_artist not in artist_to_id:
                raise SystemExit(f"unknown --mismatch-artist: {ctx_artist}")

        if args.mode == "ablation":
            context, context_mask = emb(torch.tensor([artist_to_id[artist]], device=device))
            context = torch.zeros_like(context)
        else:
            context, context_mask = emb(torch.tensor([artist_to_id[ctx_artist]], device=device))

        for idx in range(args.num_samples):
            start_ids = prompt_ids if prompt_ids else [prefix_id]
            input_ids = torch.tensor([start_ids], dtype=torch.long, device=device)
            gen_fn = generate_tokens_kv if uses_kv else generate_tokens
            with torch.no_grad():
                generated = gen_fn(model, input_ids, args.max_length, eos_id,
                                   temperature=args.temperature, top_k=args.top_k,
                                   top_p=args.top_p, context=context,
                                   context_mask=context_mask)

            full = start_ids + list(generated)
            toks = ids_to_tokens(full, tokenizer)
            if not toks or toks[0] != PREFIX_TOK:
                toks = [PREFIX_TOK] + toks
            name = f"{artist.replace(' ', '_')}_{args.mode}_{idx:02d}.mid"
            out_path = args.out_dir / name
            try:
                tokenizer.detokenize(toks).to_midi().save(str(out_path))
                logger.info(f"{name}: {len(generated)} new tokens -> {out_path}")
            except Exception as e:
                logger.warning(f"{name}: detokenize failed ({e}); skipping MIDI write")
                continue
            manifest.append({
                "file": name, "artist": artist, "context_artist": ctx_artist,
                "mode": args.mode, "sample_idx": idx,
                "prompt_midi": str(args.prompt_midi) if args.prompt_midi else None,
                "prompt_tokens": len(prompt_ids) if prompt_ids else 0,
                "generated_tokens": len(generated),
                "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                "seed": args.seed,
            })

    # Merge with any existing manifest so repeated runs into one directory
    # accumulate rather than clobber (entries are keyed by filename).
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists():
        previous = {e["file"]: e for e in json.loads(manifest_path.read_text())}
        previous.update({e["file"]: e for e in manifest})
        merged = sorted(previous.values(), key=lambda e: e["file"])
    else:
        merged = manifest
    manifest_path.write_text(json.dumps(merged, indent=1))
    logger.info(f"wrote {len(manifest)} samples; manifest now lists {len(merged)}")


if __name__ == "__main__":
    main()
