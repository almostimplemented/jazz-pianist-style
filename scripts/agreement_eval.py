#!/usr/bin/env python3
"""Sliding-window agreement: does generated music keep carrying artist identity?

The style-persistence protocol from the ISMIR 2026 paper "Learning Jazz
Pianist Style with Cross-Attention Conditioning". For each held-out
performance we take its first P tokens as a prompt, generate a continuation,
slide a 1024-token classification window along the continuation at stride
128, and record the fraction of windows the classifier attributes to the
prompt's artist. That fraction is the *agreement* with the artist, reported
as a function of window-start position.

Conditioning modes:
    conditioned   cross-attention with the true artist context
    ablation      cross-attention present, context zeroed
    mismatch      conditioned on a different artist (agreement measured
                  against both the prompt artist and the conditioning artist)

Example:
    python scripts/agreement_eval.py \
        --checkpoint-dir checkpoints/generator_best \
        --classifier checkpoints/pijama12_classifier.pt \
        --val-jsonl data/eval/pijama12_test_4096.jsonl \
        --artist-map data/eval/artist_to_id.json \
        --prompt-length 256 --out results/agreement_P256.json
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
from safetensors.torch import load_file

from aria.config import load_model_config
from aria.model import ModelConfig
from ariautils.tokenizer import AbsTokenizer

from llama_pijama.evaluation import load_model as load_classifier
from llama_pijama.models import ArtistEmbedding
from llama_pijama.utils.generation import (
    generate_tokens_kv,
    generate_tokens_kv_batched,
    tokens_to_ids,
)

logger = logging.getLogger("agreement_eval")

CLASSIFY_WINDOW = 1024
WINDOW_STRIDE = 128


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True,
                    help="Held-out sequences (4096-token JSONL)")
    ap.add_argument("--artist-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("agreement_results.json"))
    ap.add_argument("--mode", choices=["conditioned", "ablation", "mismatch"],
                    default="conditioned")
    ap.add_argument("--prompt-length", type=int, default=256,
                    help="Prompt tokens taken from each held-out performance")
    ap.add_argument("--max-continuation", type=int, default=4096)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Evaluate only the first N held-out sequences")
    ap.add_argument("--batch-size", type=int, default=1,
                    help=">1 uses batched KV-cached generation (CUDA recommended)")
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--ca-layers", type=int, nargs="*", default=[8, 9, 10, 11, 12, 13, 14, 15])
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def load_generator(args, num_artists, device):
    cfg = load_model_config(args.model_name)
    cfg.setdefault("resid_dropout", 0.0)
    cfg["grad_checkpoint"] = False
    model_config = ModelConfig(**cfg)
    from llama_pijama.models import CrossAttentionInferenceLM
    model = CrossAttentionInferenceLM(
        model_config=model_config,
        cross_attention_config={"layers": list(args.ca_layers), "dropout": 0.0})
    model.load_state_dict(load_file(args.checkpoint_dir / "model.safetensors"), strict=True)
    model = model.to(device).eval()
    model.setup_cache(batch_size=args.batch_size,
                      max_seq_len=args.prompt_length + args.max_continuation + 64,
                      dtype=torch.float32)
    emb = ArtistEmbedding(num_artists=num_artists, d_model=model_config.d_model,
                          context_length=args.context_length, dropout=0.0)
    emb.load_state_dict(load_file(args.checkpoint_dir / "artist_embeddings.safetensors"),
                        strict=True)
    return model, emb.to(device).eval()


def classify_window(classifier, token_ids, eos_id, device):
    """Classifier reads a 1024-token window; prediction is taken at the EOS position."""
    seq = list(token_ids[:CLASSIFY_WINDOW - 1]) + [eos_id]
    eos_pos = len(seq) - 1
    padded = seq + [0] * (CLASSIFY_WINDOW - len(seq))
    with torch.no_grad():
        logits = classifier(torch.tensor([padded], dtype=torch.long, device=device))
    return int(logits[0, eos_pos].argmax())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    artist_to_id = json.loads(args.artist_map.read_text())
    id_to_artist = {v: k for k, v in artist_to_id.items()}
    tokenizer = AbsTokenizer()
    eos_id = tokenizer.vocab.index(tokenizer.eos_tok)

    # Held-out prompts: first --prompt-length tokens of each sequence
    samples = []
    with jsonlines.open(args.val_jsonl) as reader:
        for rec in reader:
            artist = rec.get("metadata", {}).get("artist")
            if artist not in artist_to_id:
                continue
            ids = tokens_to_ids(rec["seq"], tokenizer)
            if len(ids) < args.prompt_length + CLASSIFY_WINDOW:
                continue
            samples.append({"artist": artist, "artist_id": artist_to_id[artist],
                            "prompt_ids": ids[:args.prompt_length]})
            if args.max_samples and len(samples) >= args.max_samples:
                break
    logger.info(f"{len(samples)} prompts at P={args.prompt_length}, mode={args.mode}")

    generator, emb = load_generator(args, len(artist_to_id), device)
    classifier, _ = load_classifier(str(args.classifier), model_name=args.model_name,
                                    num_classes=len(artist_to_id), device=device)

    # Conditioning artist per sample (differs from the prompt artist in mismatch mode)
    for s in samples:
        if args.mode == "mismatch":
            others = [i for i in id_to_artist if i != s["artist_id"]]
            s["cond_id"] = random.choice(others)
        else:
            s["cond_id"] = s["artist_id"]

    results = []
    bs = args.batch_size
    for start in range(0, len(samples), bs):
        batch = samples[start:start + bs]
        pad = bs - len(batch)
        padded = batch + [batch[0]] * pad  # KV cache is fixed-size
        input_ids = torch.tensor([s["prompt_ids"] for s in padded],
                                 dtype=torch.long, device=device)
        ctx, ctx_mask = emb(torch.tensor([s["cond_id"] for s in padded], device=device))
        if args.mode == "ablation":
            ctx = torch.zeros_like(ctx)

        with torch.no_grad():
            if bs == 1:
                gens = [generate_tokens_kv(generator, input_ids, args.max_continuation,
                                           eos_id, temperature=args.temperature,
                                           top_k=args.top_k, top_p=args.top_p,
                                           context=ctx, context_mask=ctx_mask)]
            else:
                gens = generate_tokens_kv_batched(generator, input_ids, args.max_continuation,
                                                  eos_id, temperature=args.temperature,
                                                  top_k=args.top_k, top_p=args.top_p,
                                                  context=ctx, context_mask=ctx_mask)[:len(batch)]

        for s, generated in zip(batch, gens):
            windows = []
            n = (len(generated) - CLASSIFY_WINDOW) // WINDOW_STRIDE + 1
            for w in range(max(n, 0)):
                w0 = w * WINDOW_STRIDE
                pred = classify_window(classifier, generated[w0:w0 + CLASSIFY_WINDOW],
                                       eos_id, device)
                windows.append({"position": w0, "pred": pred,
                                "correct_prompt": pred == s["artist_id"],
                                "correct_cond": pred == s["cond_id"]})
            results.append({"artist": s["artist"], "artist_id": s["artist_id"],
                            "cond_artist": id_to_artist[s["cond_id"]],
                            "generated_tokens": len(generated), "windows": windows})
        logger.info(f"{min(start + bs, len(samples))}/{len(samples)} prompts done")

    # Agreement curves: fraction of windows attributed to the artist, by position
    pos_stats = defaultdict(lambda: {"prompt": 0, "cond": 0, "total": 0})
    per_artist = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for r in results:
        for w in r["windows"]:
            st = pos_stats[w["position"]]
            st["total"] += 1
            st["prompt"] += int(w["correct_prompt"])
            st["cond"] += int(w["correct_cond"])
            pa = per_artist[r["artist"]][w["position"]]
            pa["total"] += 1
            pa["correct"] += int(w["correct_prompt"])
    positions = sorted(pos_stats)
    curve = [{"position": p,
              "agreement_prompt_artist": pos_stats[p]["prompt"] / pos_stats[p]["total"],
              "agreement_cond_artist": pos_stats[p]["cond"] / pos_stats[p]["total"],
              "n_windows": pos_stats[p]["total"]} for p in positions]
    mean_agreement = (sum(c["agreement_prompt_artist"] for c in curve) / len(curve)) if curve else 0.0
    per_artist_mean = {a: sum(d[p]["correct"] for p in d) / max(sum(d[p]["total"] for p in d), 1)
                       for a, d in per_artist.items()}

    out = {
        "config": {"mode": args.mode, "prompt_length": args.prompt_length,
                   "classify_window": CLASSIFY_WINDOW, "window_stride": WINDOW_STRIDE,
                   "max_continuation": args.max_continuation, "seed": args.seed,
                   "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                   "n_prompts": len(results)},
        "mean_agreement": mean_agreement,
        "agreement_curve": curve,
        "per_artist_mean_agreement": per_artist_mean,
        "per_sample": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    logger.info(f"mean agreement (positional): {mean_agreement:.4f} over {len(curve)} positions")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
