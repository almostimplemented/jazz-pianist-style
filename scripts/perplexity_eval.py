#!/usr/bin/env python3
"""Token-weighted test-set perplexity (Table 2 of the paper).

Teacher-forced cross-entropy over every predicted token of the 4096-token test
sequences, summed and divided by the token count (not a mean of per-batch
means), plus a per-artist breakdown. Three model types:

    cross_attention  the conditioned generator (artist context from its
                     learned embeddings)
    baseline         Aria fine-tuned on the same data without cross-attention
    pretrained       Aria as released, no fine-tuning

The paper's run used fp16 autocast on CUDA; elsewhere this runs in fp32, which
can move the last digit.

Example:
    python scripts/perplexity_eval.py --model-type cross_attention \
        --checkpoint checkpoints/generator \
        --test-jsonl data/pijama12_4096/test.jsonl \
        --artist-map data/pijama12_4096/artist_to_id.json \
        --out results/perplexity_cross_attention.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from aria.config import load_model_config
from aria.model import ModelConfig, TransformerLM
from ariautils.tokenizer import AbsTokenizer

from llama_pijama.models import ArtistEmbedding, CrossAttentionTransformerLM
from llama_pijama.training.cross_attention_dataset import CrossAttentionDataset

logger = logging.getLogger("perplexity_eval")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-type", choices=["cross_attention", "baseline", "pretrained"],
                    required=True)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="generator directory (cross_attention) or a .safetensors file / "
                         "directory holding model.safetensors")
    ap.add_argument("--test-jsonl", type=Path, required=True)
    ap.add_argument("--artist-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/perplexity.json"))
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--ca-layers", type=int, nargs="+", default=list(range(8, 16)))
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=None, help="smoke tests")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def load_state(path: Path):
    f = path / "model.safetensors" if path.is_dir() else path
    return {k.replace("_orig_mod.", ""): v for k, v in load_file(str(f)).items()}


def load_model(args, num_artists: int, device: str):
    cfg = load_model_config(args.model_name)
    cfg.setdefault("resid_dropout", 0.0)
    cfg["grad_checkpoint"] = False
    model_config = ModelConfig(**cfg)
    if args.model_type == "cross_attention":
        model = CrossAttentionTransformerLM(
            model_config, {"layers": list(args.ca_layers), "dropout": 0.0, "gate_init": 0.1})
        model.load_state_dict(load_state(args.checkpoint), strict=True)
        emb = ArtistEmbedding(num_artists=num_artists, d_model=model_config.d_model,
                              context_length=args.context_length, dropout=0.0)
        emb.load_state_dict(load_file(str(args.checkpoint / "artist_embeddings.safetensors")),
                            strict=True)
        return model.to(device).eval(), emb.to(device).eval()
    model = TransformerLM(model_config)
    missing, unexpected = model.load_state_dict(load_state(args.checkpoint), strict=False)
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
    if unexpected:
        logger.info(f"ignoring {len(unexpected)} checkpoint keys not used by TransformerLM")
    return model.to(device).eval(), None


@torch.no_grad()
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    artist_to_id = json.loads(args.artist_map.read_text())
    id_to_artist = {v: k for k, v in artist_to_id.items()}

    dataset = CrossAttentionDataset(jsonl_path=str(args.test_jsonl), artist_to_id=artist_to_id,
                                    tokenizer=AbsTokenizer(), max_seq_len=args.max_seq_len)
    if args.max_samples:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"{len(dataset)} test sequences on {device}")

    model, emb = load_model(args, len(artist_to_id), device)
    use_amp = device == "cuda"

    loss_sum, token_count = 0.0, 0
    per_artist = defaultdict(lambda: [0.0, 0])
    for bi, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        artist_ids = batch["artist_id"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            if emb is not None:
                context, context_mask = emb(artist_ids)
                logits = model(input_ids, context=context, context_mask=context_mask)
            else:
                logits = model(input_ids)
        for i in range(input_ids.size(0)):
            lab = labels[i].view(-1)
            n = int((lab != -100).sum())
            if n == 0:
                continue
            s = float(F.cross_entropy(logits[i].float().view(-1, logits.size(-1)), lab,
                                      ignore_index=-100, reduction="sum"))
            loss_sum += s
            token_count += n
            a = id_to_artist[int(artist_ids[i])]
            per_artist[a][0] += s
            per_artist[a][1] += n
        if (bi + 1) % 10 == 0:
            logger.info(f"  {bi + 1}/{len(loader)} batches")

    nll = loss_sum / max(token_count, 1)
    out = {
        "model_type": args.model_type, "checkpoint": str(args.checkpoint),
        "test_jsonl": str(args.test_jsonl), "num_sequences": len(dataset),
        "num_tokens": token_count, "test_nll": nll, "test_perplexity": math.exp(min(nll, 20)),
        "per_artist": {a: {"test_nll": s / n, "test_perplexity": math.exp(min(s / n, 20)),
                           "num_tokens": n} for a, (s, n) in sorted(per_artist.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    logger.info(f"NLL {nll:.4f}  perplexity {out['test_perplexity']:.2f}  "
                f"over {token_count} tokens -> {args.out}")


if __name__ == "__main__":
    main()
