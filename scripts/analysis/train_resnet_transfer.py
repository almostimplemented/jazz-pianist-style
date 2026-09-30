#!/usr/bin/env python3
"""Train Cheston et al.'s ResNet-50 (their architecture + recipe) on OUR data.

Arms:
  --arm real   train on real PiJAMA-12 train clips, select on real val
  --arm synth  train on clean-room synthetic clips ONLY, select on SYNTHETIC val
               (real data touches nothing until the final test eval)

Recipe mirrors config/baselines/resnet50-jtd+pijama-augment.yaml from their
repo: SGD lr=0.01 momentum=0.9 wd=1e-4, cosine schedule, batch 5, CE loss,
augmentation (their augment_midi: transpose +/-6, dilate 0.2, velocity 12)
with p=0.5 and clip-start jitter at train time only.

Eval: --eval-checkpoint scores the real test split: clip top-1, and track
accuracy by majority vote (ties broken by mean logit; the paper's number) and
by mean softmax.

Everything runs locally (MPS/CPU); no cluster dependency.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import types
from collections import defaultdict
from pathlib import Path

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get("RESNET_DATA_DIR", REPO_ROOT / "data" / "resnet_transfer"))
_ARTIST_TO_ID = None


def artist_to_id():
    """Artist -> class index, read from the clip dataset directory."""
    global _ARTIST_TO_ID
    if _ARTIST_TO_ID is None:
        _ARTIST_TO_ID = json.loads((BASE / "artist_to_id.json").read_text())
    return _ARTIST_TO_ID
CLIP_LEN = 30.0


sys.path.insert(0, str(REPO_ROOT))
from llama_pijama.external.cheston_dpi import extractors as EX
from llama_pijama.external.cheston_dpi import resnet as RESNET


def clip_starts(duration: float):
    starts = list(np.arange(0.0, max(duration - 15.0, 1.0), CLIP_LEN))
    return starts or [0.0]


class ClipDataset(Dataset):
    def __init__(self, index_csv: Path, augment: bool):
        self.augment = augment
        self.items = []  # (midi_path, start, label_id, group)
        with open(index_csv) as f:
            for row in csv.DictReader(f):
                label = artist_to_id()[row["label"]]
                for s in clip_starts(float(row["duration_s"])):
                    self.items.append((str(BASE / row["file"]), s, label, row["group"]))
        self._pm_cache = {}

    def __len__(self):
        return len(self.items)

    def _load_pm(self, path):
        from pretty_midi import PrettyMIDI
        if path not in self._pm_cache:
            if len(self._pm_cache) > 64:
                self._pm_cache.clear()
            self._pm_cache[path] = PrettyMIDI(path)
        return self._pm_cache[path]

    def __getitem__(self, i):
        path, start, label, _group = self.items[i]
        try:
            pm = self._load_pm(path)
            if self.augment:
                start = max(0.0, start + random.uniform(-2.0, 2.0))  # jitter_start
            clip_pm = EX.RollExtractor(pm, clip_start=start).output_midi
            if self.augment and random.random() < 0.5:
                clip_pm = EX.augment_midi(clip_pm)
            roll = EX.normalize_array(EX.RollExtractor(clip_pm, clip_start=0.0).roll)
        except Exception:
            return None
        return torch.tensor(roll, dtype=torch.float32)[None], label, i


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    x = torch.stack([b[0] for b in batch])
    y = torch.tensor([b[1] for b in batch])
    idx = torch.tensor([b[2] for b in batch])
    return x, y, idx


def majority_vote(logit_rows):
    """Most-voted class over a track's clips; exact vote ties go to the tied
    class with the highest mean logit (the paper-wide convention)."""
    logits = np.stack(logit_rows)
    votes = np.bincount(logits.argmax(1), minlength=logits.shape[1])
    tied = np.flatnonzero(votes == votes.max())
    return int(tied[logits[:, tied].mean(0).argmax()])


@torch.no_grad()
def evaluate(model, ds, device, batch_size, max_batches=None):
    """Clip accuracy, plus track accuracy by majority vote (the paper's number)
    and by mean softmax."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, collate_fn=collate)
    correct = total = 0
    track_logits = defaultdict(list)
    track_label = {}
    for bi, batch in enumerate(dl):
        if batch is None:
            continue
        if max_batches and bi >= max_batches:
            break
        x, y, idx = batch
        logits = model(x.to(device)).float().cpu()
        correct += int((logits.argmax(1) == y).sum())
        total += len(y)
        for lg, yy, ii in zip(logits, y, idx):
            g = ds.items[int(ii)][3]
            track_logits[g].append(lg.numpy())
            track_label[g] = int(yy)
    model.train()
    n_tracks = max(len(track_logits), 1)
    softmax = lambda a: np.exp(a - a.max(1, keepdims=True)) / np.exp(a - a.max(1, keepdims=True)).sum(1, keepdims=True)
    return {
        "clip_acc": correct / max(total, 1),
        "track_acc_majority_vote": sum(majority_vote(v) == track_label[g]
                                       for g, v in track_logits.items()) / n_tracks,
        "track_acc_mean_softmax": sum(int(softmax(np.stack(v)).mean(0).argmax() == track_label[g])
                                      for g, v in track_logits.items()) / n_tracks,
        "n_clips": total, "n_tracks": len(track_logits),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["real", "synth"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=5)   # their recipe
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-checkpoint", type=Path, default=None)
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="checkpoint to resume weights from")
    ap.add_argument("--start-epoch", type=int, default=0,
                    help="epoch to resume at (cosine schedule fast-forwarded)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = RESNET.ResNet50(num_classes=len(artist_to_id())).to(device)

    if args.eval_checkpoint:
        sd = torch.load(args.eval_checkpoint, map_location="cpu")
        model.load_state_dict(sd["model"])
        test_ds = ClipDataset(BASE / "real_test_index.csv", augment=False)
        m = evaluate(model, test_ds, device, args.batch_size)
        result = {"checkpoint": str(args.eval_checkpoint),
                  **{f"test_{k}" if "acc" in k else k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in m.items()}}
        print(json.dumps(result))
        out = args.eval_checkpoint.parent / "test_eval.json"
        out.write_text(json.dumps(result, indent=1))
        return

    assert args.arm, "--arm required for training"
    out_dir = args.out_dir or (BASE / f"run_{args.arm}")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = ClipDataset(BASE / f"{args.arm}_train_index.csv", augment=True)
    val_ds = ClipDataset(BASE / f"{args.arm}_val_index.csv", augment=False)
    print(f"arm={args.arm} device={device} train_clips={len(train_ds)} val_clips={len(val_ds)}")

    best_val = 0.0
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(ck["model"])
        best_val = float(ck.get("val_clip_acc", 0.0))
        print(f"resumed weights from {args.resume_from} (val_clip {best_val:.3f}); "
              f"restarting at epoch {args.start_epoch}; note: optimizer momentum resets")
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          nesterov=False, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    for _ in range(args.start_epoch):
        sched.step()
    loss_fn = torch.nn.CrossEntropyLoss()
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, collate_fn=collate, drop_last=True)

    log_path = out_dir / "log.json"
    log = json.loads(log_path.read_text()) if (args.resume_from and log_path.exists()) else []
    log = [e for e in log if e["epoch"] < args.start_epoch]
    for epoch in range(args.start_epoch, args.epochs):
        run_loss = n_batches = 0
        for batch in dl:
            if batch is None:
                continue
            x, y, _ = batch
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            run_loss += float(loss)
            n_batches += 1
        sched.step()
        vm = evaluate(model, val_ds, device, args.batch_size)
        val_clip, val_track = vm["clip_acc"], vm["track_acc_majority_vote"]
        entry = {"epoch": epoch, "train_loss": round(run_loss / max(n_batches, 1), 4),
                 "val_clip_acc": round(val_clip, 4), "val_track_acc": round(val_track, 4),
                 "lr": sched.get_last_lr()[0]}
        log.append(entry)
        print(json.dumps(entry), flush=True)
        if val_clip > best_val:
            best_val = val_clip
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_clip_acc": val_clip}, out_dir / "best.pt")
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))
    torch.save({"model": model.state_dict(), "epoch": args.epochs - 1}, out_dir / "final.pt")


if __name__ == "__main__":
    main()
