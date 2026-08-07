#!/usr/bin/env python3
"""Train the gated cross-attention conditional generator (single GPU, local).

Trains the conditional generator from the ISMIR 2026 paper "Learning Jazz
Pianist Style with Cross-Attention Conditioning". Defaults reproduce the
paper's training run (cross-attention on the last 8 of 16 layers,
4-vector artist context, 15 epochs at effective batch size 32, peak LR 5e-6
with 0.5-epoch linear warmup then cosine decay, fp16 on CUDA, seed 42).

Inputs are local JSONL files of pre-tokenized sequences (see data/README.md)
plus the pretrained Aria medium generation checkpoint (safetensors).

Example:
    python scripts/train_cross_attention.py \
        --train-jsonl data/train.jsonl --val-jsonl data/val.jsonl \
        --pretrained-checkpoint checkpoints/aria-medium-gen.safetensors \
        --out-dir checkpoints/run1

A full paper-scale run needs a >=40GB GPU for ~18h. For a quick smoke test:
    ... --epochs 1 --max-steps 20 --batch-size 1
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.amp import autocast
from torch.utils.data import DataLoader

from aria.config import load_model_config
from aria.model import ModelConfig, TransformerLM
from ariautils.tokenizer import AbsTokenizer

from llama_pijama.models import ArtistEmbedding, CrossAttentionTransformerLM
from llama_pijama.training.cross_attention_dataset import (
    CrossAttentionDataset,
    build_artist_mapping,
)
from llama_pijama.utils.generation import generate_tokens

logger = logging.getLogger("train_cross_attention")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--pretrained-checkpoint", type=Path, default=None,
                    help="Aria medium generation weights (.safetensors). "
                         "Omit to train from random init (NOT the paper setting).")
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints/cross_attention"))
    # Paper hyperparameters (camera-ready run cleaned-cross-at)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--warmup-epochs", type=float, default=0.5)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16",
                    help="AMP dtype; only active on CUDA")
    ap.add_argument("--ca-layers", type=int, nargs="*", default=[8, 9, 10, 11, 12, 13, 14, 15])
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--ca-dropout", type=float, default=0.1)
    ap.add_argument("--gate-init", type=float, default=0.1)
    ap.add_argument("--embedding-dropout", type=float, default=0.3)
    ap.add_argument("--conditioning-dropout", type=float, default=0.3,
                    help="Per-sample probability of zeroing the artist context "
                         "during training (unconditional branch for CFG-style use)")
    ap.add_argument("--model-name", default="medium", help="Aria config name")
    ap.add_argument("--model-config-json", type=Path, default=None,
                    help="JSON file overriding the model config (tiny/debug runs)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Stop after N optimizer steps per epoch (smoke tests)")
    ap.add_argument("--sample-every-n-epochs", type=int, default=0,
                    help="Generate short monitoring samples every N epochs (0 = off)")
    ap.add_argument("--no-cross-attention", action="store_true",
                    help="Train the unconditioned ablation baseline instead")
    return ap.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(args, num_artists: int, device: torch.device):
    if args.model_config_json:
        cfg_dict = json.loads(args.model_config_json.read_text())
        return _build_from_cfg(args, cfg_dict, num_artists, device)
    try:
        cfg_dict = load_model_config(args.model_name)
        cfg_dict.setdefault("resid_dropout", 0.0)
    except Exception as e:  # aria config lookup can fail on odd installs
        logger.warning(f"load_model_config({args.model_name}) failed ({e}); using medium defaults")
        cfg_dict = dict(d_model=1536, n_heads=24, n_layers=16, ff_mult=4, drop_p=0.0,
                        max_seq_len=8192, vocab_size=17727, grad_checkpoint=True,
                        resid_dropout=0.0)

    return _build_from_cfg(args, cfg_dict, num_artists, device)


def _build_from_cfg(args, cfg_dict, num_artists, device):
    ca_config = None if args.no_cross_attention else {
        "layers": list(args.ca_layers),
        "dropout": args.ca_dropout,
        "gate_init": args.gate_init,
    }
    model = CrossAttentionTransformerLM(ModelConfig(**cfg_dict), ca_config)

    artist_embeddings = None
    if not args.no_cross_attention:
        artist_embeddings = ArtistEmbedding(
            num_artists=num_artists,
            d_model=cfg_dict["d_model"],
            context_length=args.context_length,
            dropout=args.embedding_dropout,
        )

    if args.pretrained_checkpoint:
        logger.info(f"Loading pretrained weights from {args.pretrained_checkpoint}")
        base = TransformerLM(model.model_config)
        sd = load_file(str(args.pretrained_checkpoint)) \
            if args.pretrained_checkpoint.suffix == ".safetensors" \
            else torch.load(str(args.pretrained_checkpoint), map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        base.load_state_dict(sd, strict=True)
        missing, unexpected = model.load_pretrained_weights(base.state_dict())
        new_ca = [k for k in missing if "cross_attention" in k or "norm_cross" in k]
        other = [k for k in missing if k not in new_ca]
        if other:
            logger.warning(f"Missing non-CA keys: {other[:10]} (total {len(other)})")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:10]} (total {len(unexpected)})")
        logger.info(f"Initialized {len(new_ca)} new cross-attention parameters")

    model = model.to(device)
    if artist_embeddings is not None:
        artist_embeddings = artist_embeddings.to(device)
    total = sum(p.numel() for p in model.parameters())
    ca = sum(p.numel() for n, p in model.named_parameters()
             if "cross_attention" in n or "norm_cross" in n)
    logger.info(f"Params: total={total:,} cross_attn={ca:,}"
                + (f" artist_embed={sum(p.numel() for p in artist_embeddings.parameters()):,}"
                   if artist_embeddings else ""))
    return model, artist_embeddings


def compute_loss(model, artist_embeddings, batch, device, cond_drop: float,
                 amp_enabled: bool, amp_dtype):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    artist_ids = batch.get("artist_id", torch.zeros(input_ids.size(0), dtype=torch.long)).to(device)

    with autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        if artist_embeddings is not None:
            context, context_mask = artist_embeddings(artist_ids)
            # Conditioning dropout: zero the *returned* context (projection has
            # bias + LayerNorm, so zeroed embedding rows would not give zero
            # context). A zero context row yields exactly zero cross-attention
            # output — the same path evaluation uses for the ablation mode.
            if cond_drop > 0.0 and model.training:
                keep = torch.rand(context.size(0), device=context.device) >= cond_drop
                context = context * keep.to(context.dtype).view(-1, 1, 1)
        else:
            context, context_mask = None, None
        logits = model(input_ids, context=context, context_mask=context_mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                               ignore_index=-100)
    return loss


@torch.no_grad()
def validate(model, artist_embeddings, val_loader, device, id_to_artist,
             amp_enabled, amp_dtype) -> float:
    model.eval()
    if artist_embeddings is not None:
        artist_embeddings.eval()
    total = 0.0
    per_artist = defaultdict(list)
    for batch in val_loader:
        loss = compute_loss(model, artist_embeddings, batch, device, 0.0,
                            amp_enabled, amp_dtype)
        total += loss.item()
        # per-sample per-artist breakdown
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        artist_ids = batch.get("artist_id", torch.zeros(input_ids.size(0), dtype=torch.long)).to(device)
        if id_to_artist and artist_embeddings is not None:
            context, context_mask = artist_embeddings(artist_ids)
            with autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits = model(input_ids, context=context, context_mask=context_mask)
            for i in range(input_ids.size(0)):
                l = F.cross_entropy(logits[i].view(-1, logits.size(-1)),
                                    labels[i].view(-1), ignore_index=-100)
                per_artist[id_to_artist.get(int(artist_ids[i]), "?")].append(l.item())
    avg = total / max(len(val_loader), 1)
    if per_artist:
        logger.info(f"  {'Artist':<20} {'Val loss':>9} {'PPL':>9} {'N':>4}")
        for artist in sorted(per_artist):
            al = sum(per_artist[artist]) / len(per_artist[artist])
            logger.info(f"  {artist:<20} {al:>9.4f} {min(math.exp(al), 1e6):>9.1f} "
                        f"{len(per_artist[artist]):>4}")
    model.train()
    if artist_embeddings is not None:
        artist_embeddings.train()
    return avg


def save_checkpoint(out_dir: Path, tag: str, model, artist_embeddings, args,
                    epoch: int, best_val_loss: float):
    ckpt = out_dir / tag
    ckpt.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), ckpt / "model.safetensors")
    if artist_embeddings is not None:
        save_file(artist_embeddings.state_dict(), ckpt / "artist_embeddings.safetensors")
    # Clean, secret-free run config: only reproducibility-relevant fields.
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    meta.update(epoch=epoch, best_val_loss=best_val_loss)
    (ckpt / "config.json").write_text(json.dumps(meta, indent=1))
    gates = {i: float(l.cross_attention.gate)
             for i, l in enumerate(model.model.encode_layers)
             if getattr(l, "use_cross_attention", False)}
    if gates:
        logger.info("Gates at save: " + ", ".join(f"L{k}:{v:.4f}" for k, v in gates.items()))
    logger.info(f"Saved {tag} checkpoint to {ckpt}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device()
    amp_enabled = args.mixed_precision != "no" and device.type == "cuda"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "no": torch.float32}[args.mixed_precision]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    logger.info(f"device={device} amp={'off' if not amp_enabled else args.mixed_precision}")

    tokenizer = AbsTokenizer()
    artist_to_id = build_artist_mapping([str(args.train_jsonl), str(args.val_jsonl)])
    id_to_artist = {v: k for k, v in artist_to_id.items()}
    logger.info(f"{len(artist_to_id)} artists")

    model, artist_embeddings = build_model(args, len(artist_to_id), device)

    train_ds = CrossAttentionDataset(str(args.train_jsonl), artist_to_id, tokenizer,
                                     max_seq_len=args.max_seq_len, cache_size=10000)
    val_ds = CrossAttentionDataset(str(args.val_jsonl), artist_to_id, tokenizer,
                                   max_seq_len=args.max_seq_len, cache_size=1000)
    gen = torch.Generator(); gen.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              generator=gen, worker_init_fn=_seed_worker)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    params = list(model.parameters())
    if artist_embeddings is not None:
        params += list(artist_embeddings.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    if warmup_steps > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        scheduler = SequentialLR(
            optimizer,
            [LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps),
             CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                               eta_min=args.lr * 0.01)],
            milestones=[warmup_steps])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.lr * 0.01)

    best_val = float("inf")
    log_path = args.out_dir / "log.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        if artist_embeddings is not None:
            artist_embeddings.train()
        running = 0.0
        opt_steps = 0
        for batch_idx, batch in enumerate(train_loader):
            loss = compute_loss(model, artist_embeddings, batch, device,
                                args.conditioning_dropout, amp_enabled, amp_dtype)
            running += loss.item()
            (scaler.scale(loss) if scaler.is_enabled() else loss).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_steps += 1
                if args.max_steps and opt_steps >= args.max_steps:
                    break
            if batch_idx % 100 == 0:
                logger.info(f"epoch {epoch} batch {batch_idx}/{len(train_loader)} "
                            f"loss {loss.item():.4f} lr {optimizer.param_groups[0]['lr']:.2e}")

        val_loss = validate(model, artist_embeddings, val_loader, device,
                            id_to_artist, amp_enabled, amp_dtype)
        avg_train = running / max(batch_idx + 1, 1)
        logger.info(f"epoch {epoch}: train {avg_train:.4f} val {val_loss:.4f} "
                    f"({time.time()-t0:.0f}s)")
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": avg_train,
                                "val_loss": val_loss,
                                "lr": optimizer.param_groups[0]["lr"]}) + "\n")

        save_checkpoint(args.out_dir, "latest", model, artist_embeddings, args, epoch, best_val)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args.out_dir, "best", model, artist_embeddings, args, epoch, best_val)

        if args.sample_every_n_epochs and epoch % args.sample_every_n_epochs == 0 \
                and artist_embeddings is not None:
            model.eval()
            eos_id = tokenizer.vocab.index(tokenizer.eos_tok)
            prefix_id = tokenizer.vocab.index(("prefix", "instrument", "piano"))
            name, aid = next(iter(artist_to_id.items()))
            ctx, mask = artist_embeddings(torch.tensor([aid], device=device))
            out = generate_tokens(model, torch.tensor([[prefix_id]], device=device),
                                  256, eos_id, temperature=0.95, context=ctx, context_mask=mask)
            logger.info(f"sample for {name}: {len(out)} tokens")
            model.train()


if __name__ == "__main__":
    main()
