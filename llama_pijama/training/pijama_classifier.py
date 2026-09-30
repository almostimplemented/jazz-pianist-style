import os
import json
import logging
import time
import accelerate
from logging.handlers import RotatingFileHandler

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import mmap
import jsonlines

from ariautils.tokenizer import AbsTokenizer
from transformers import get_cosine_schedule_with_warmup


def _setup_logger(project_dir: str):
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    # Allow messages to flow to root handlers configured by entrypoints
    logger.propagate = True
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(name)s: [%(levelname)s] %(message)s")
    fh = RotatingFileHandler(os.path.join(project_dir, "logs.txt"), backupCount=5, maxBytes=1024**3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Add console handler only if root has no handlers
    if not logging.getLogger().handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logging.getLogger(__name__)


def _setup_project_dir(project_dir: str | None):
    if not project_dir:
        if not os.path.isdir("./experiments"):
            os.mkdir("./experiments")
        names = [d for d in os.listdir("./experiments") if os.path.isdir(os.path.join("experiments", d))]
        idx = 0
        while str(idx) in names:
            idx += 1
        project_dir_abs = os.path.abspath(os.path.join("experiments", str(idx)))
        os.mkdir(project_dir_abs)
    else:
        if os.path.isdir(project_dir):
            # Directory exists - use it (normal case with Hydra or existing dir)
            project_dir_abs = os.path.abspath(project_dir)
        elif os.path.isfile(project_dir):
            raise FileExistsError("The provided path points toward an existing file")
        else:
            # Create directory if it doesn't exist
            os.makedirs(project_dir, exist_ok=True)
            project_dir_abs = os.path.abspath(project_dir)

    # Ensure subdirectories exist
    os.makedirs(os.path.join(project_dir_abs, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(project_dir_abs, "logs"), exist_ok=True)
    return project_dir_abs


class JsonlClassificationDataset(Dataset):
    def __init__(self, jsonl_path: str, tag_to_id: dict, max_seq_len: int):
        self.jsonl_path = jsonl_path
        self.tag_to_id = tag_to_id
        self.max_seq_len = max_seq_len
        self.tokenizer = AbsTokenizer()
        self.index = []
        self._transform = None
        self._f = open(jsonl_path, "rb")
        self._m = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        while True:
            pos = self._m.tell()
            line = self._m.readline()
            if not line:
                break
            self.index.append(pos)

    def set_transform(self, transform, prob: float = 1.0):
        import random as _random
        if isinstance(transform, list):
            def _combined_transform(x):
                for fn in transform:
                    x = fn(x)
                return x
            base_fn = _combined_transform
        else:
            base_fn = transform

        if prob < 1.0:
            def _prob_transform(x):
                return base_fn(x) if _random.random() < prob else x
            self._transform = _prob_transform
        else:
            self._transform = base_fn

    def __getitem__(self, idx: int):
        self._m.seek(self.index[idx])
        raw = self._m.readline().decode("utf-8")
        rec = json.loads(raw)
        seq = rec["seq"]

        # Convert JSON lists back to tuples for hashability in tokenizer.encode
        def _format(tok):
            return tuple(tok) if isinstance(tok, list) else tok

        seq = [_format(tok) for tok in seq]

        if self._transform:
            seq = self._transform(seq)
        seq = seq[: self.max_seq_len]
        if self.tokenizer.eos_tok not in seq:
            seq[-1] = self.tokenizer.eos_tok
        eos_index = seq.index(self.tokenizer.eos_tok)
        pos_tensor = torch.tensor(eos_index)
        seq = seq + [self.tokenizer.pad_tok] * (self.max_seq_len - len(seq))
        enc = self.tokenizer.encode(seq)
        seq_tensor = torch.tensor(enc)
        tag = rec["metadata"]["artist"]
        assert tag in self.tag_to_id, f"Unknown tag: {tag}"
        tag_tensor = torch.tensor(self.tag_to_id[tag])
        return seq_tensor, pos_tensor, tag_tensor

    def __len__(self):
        return len(self.index)

    def get_class_counts(self) -> dict:
        """Count samples per class for computing class weights."""
        counts = {tag: 0 for tag in self.tag_to_id}
        self._m.seek(0)
        for _ in range(len(self.index)):
            line = self._m.readline().decode("utf-8")
            rec = json.loads(line)
            tag = rec["metadata"]["artist"]
            if tag in counts:
                counts[tag] += 1
        return counts


def _get_optim(
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
    grad_acc_steps: int,
    freeze_base: bool = False,
):
    # Higher LR and lower weight decay when only training the classifier head
    lr = 1e-3 if freeze_base else 1e-5
    weight_decay = 1e-2 if freeze_base else 0.1

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-5
    )

    # Scheduler aware of gradient accumulation with warmup + cosine decay
    opt_steps_per_epoch = (steps_per_epoch + grad_acc_steps - 1) // grad_acc_steps
    total_opt_steps = max(1, num_epochs * opt_steps_per_epoch)
    warmup_steps = max(1, int(0.05 * total_opt_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )
    return optimizer, scheduler


def _prepare_loaders(train_path: str, val_path: str, tag_to_id: dict, max_seq_len: int, batch_size: int, num_workers: int, use_augmentation: bool = True, augmentation_prob: float = 0.5):
    train_ds = JsonlClassificationDataset(train_path, tag_to_id, max_seq_len)
    val_ds = JsonlClassificationDataset(val_path, tag_to_id, max_seq_len)

    # Apply data augmentation to training set
    if use_augmentation:
        augmentations = AbsTokenizer().export_data_aug()
        train_ds.set_transform(augmentations, prob=augmentation_prob)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # Get class counts for balanced loss
    class_counts = train_ds.get_class_counts()
    return train_loader, val_loader, class_counts


def _compute_class_weights(class_counts: dict, tag_to_id: dict, device: str = "cpu") -> torch.Tensor:
    """Compute inverse frequency class weights for balanced training."""
    num_classes = len(tag_to_id)
    counts = torch.zeros(num_classes)
    for tag, idx in tag_to_id.items():
        counts[idx] = class_counts.get(tag, 1)
    # Inverse frequency weighting, normalized
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # Scale so mean weight = 1
    return weights.to(device)


def train(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    artist_to_id_path: str,
    num_epochs: int = 10,
    batch_size: int = 16,
    num_workers: int = 2,
    grad_acc_steps: int = 1,
    project_dir: str | None = None,
    checkpoint_path: str | None = None,
    freeze_base: bool = False,
    use_pretrained: bool = True,
    use_wandb: bool = False,
    balanced_loss: bool = True,
    use_augmentation: bool = True,
    augmentation_prob: float = 0.5,
    early_stopping_patience: int = 5,
    run_name: str | None = None,
):
    # Lazy imports so that dataset can be imported without aria installed
    from aria.config import load_model_config
    from aria.utils import _load_weight
    from aria.model import TransformerCL, ModelConfig

    with open(artist_to_id_path, "r") as f:
        artist_to_id = json.load(f)

    tokenizer = AbsTokenizer()
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.class_size = len(artist_to_id)
    # Set max_seq_len to 1024 to reduce compute (default is 8192)
    model_config.max_seq_len = 1024
    # Add residual dropout for regularization (Aria paper uses 0.0-0.2 linearly, pianist model uses 0.2)
    model_config.resid_dropout = 0.2
    max_seq_len = model_config.max_seq_len

    accelerator = accelerate.Accelerator(project_dir=project_dir, gradient_accumulation_steps=grad_acc_steps)

    if accelerator.is_main_process:
        project_dir = _setup_project_dir(project_dir)
        logger = _setup_logger(project_dir)
    else:
        project_dir = project_dir or "./experiments"
        logger = logging.getLogger(__name__)

    logger.info(f"Project directory: {project_dir}")
    logger.info(f"Artists: {len(artist_to_id)}")
    logger.info(f"Training config: epochs={num_epochs}, batch_size={batch_size}, workers={num_workers}")

    # Initialize W&B
    wandb_run = None
    if use_wandb and accelerator.is_main_process:
        import wandb
        if not run_name:
            run_name = f"clf-bs{batch_size}-ga{grad_acc_steps}-{'frozen' if freeze_base else 'full'}-{os.path.basename(project_dir)}"
        wandb_run = wandb.init(
            project="llama-pijama",
            name=run_name,
            config={
                "model_name": model_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "grad_acc_steps": grad_acc_steps,
                "effective_batch_size": batch_size * grad_acc_steps,
                "num_workers": num_workers,
                "max_seq_len": max_seq_len,
                "num_classes": len(artist_to_id),
                "freeze_base": freeze_base,
                "use_pretrained": use_pretrained,
                "checkpoint_path": checkpoint_path,
                "balanced_loss": balanced_loss,
                "use_augmentation": use_augmentation,
                "augmentation_prob": augmentation_prob,
                "early_stopping_patience": early_stopping_patience,
            }
        )
        logger.info(f"W&B run initialized: {run_name}")

    model = TransformerCL(model_config)

    # Load pretrained weights
    if use_pretrained:
        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state = _load_weight(checkpoint_path)
        else:
            # Download from HuggingFace Hub
            from huggingface_hub import hf_hub_download
            logger.info(f"Downloading pretrained aria-{model_name}-base from HuggingFace Hub")
            hf_checkpoint = hf_hub_download(
                repo_id=f"loubb/aria-{model_name}-base",
                filename="model-gen.safetensors"
            )
            logger.info(f"Loading checkpoint from {hf_checkpoint}")
            state = _load_weight(hf_checkpoint)

        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        logger.info("Pretrained weights loaded successfully")

        # Reinitialize pad token embedding
        torch.nn.init.normal_(model.model.tok_embeddings.weight.data[1:2], mean=0.0, std=0.02)
    else:
        logger.info("Training from random initialization")

    # Optionally freeze base model (only train classifier head)
    if freeze_base:
        logger.info("Freezing base model parameters - only training classifier head")
        for name, param in model.named_parameters():
            if not name.startswith("class_head"):
                param.requires_grad = False
        # Log trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    train_loader, val_loader, class_counts = _prepare_loaders(
        train_path=train_data_path,
        val_path=val_data_path,
        tag_to_id=artist_to_id,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        use_augmentation=use_augmentation,
        augmentation_prob=augmentation_prob,
    )
    if use_augmentation:
        logger.info(f"Data augmentation enabled (prob={augmentation_prob}): pitch (±5 semitones), tempo (±20%), velocity")

    # Log class distribution
    logger.info("Class distribution:")
    for artist, count in sorted(class_counts.items(), key=lambda x: -x[1])[:5]:
        logger.info(f"  {artist}: {count} samples")
    logger.info(f"  ... and {len(class_counts) - 5} more artists")

    optimizer, scheduler = _get_optim(
        model=model,
        num_epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        grad_acc_steps=grad_acc_steps,
        freeze_base=freeze_base,
    )

    (model, train_loader, val_loader, optimizer, scheduler) = accelerator.prepare(
        model, train_loader, val_loader, optimizer, scheduler
    )

    # Create loss function with optional class weighting
    if balanced_loss:
        class_weights = _compute_class_weights(class_counts, artist_to_id, device=accelerator.device)
        logger.info(f"Using balanced loss with class weights (min={class_weights.min():.3f}, max={class_weights.max():.3f})")
        # Log full class weight vector
        id_to_artist_init = {v: k for k, v in artist_to_id.items()}
        for i in range(len(class_weights)):
            logger.info(f"  {id_to_artist_init[i]}: weight={class_weights[i]:.3f}")
        if wandb_run:
            wandb_run.config.update({
                "class_weights": {id_to_artist_init[i]: float(class_weights[i]) for i in range(len(class_weights))}
            })
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        logger.info("Using standard unweighted CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss()

    def train_epoch(epoch: int):
        model.train()
        avg_loss = 0.0
        buffer = []
        epoch_start = time.time()
        from tqdm import tqdm

        for step, batch in (pbar := tqdm(enumerate(train_loader), total=len(train_loader), leave=False)):
            seqs, eos_pos, labels = batch
            with accelerator.accumulate(model):
                logits = model(seqs)
                logits = logits[torch.arange(logits.shape[0], device=logits.device), eos_pos]
                loss = loss_fn(logits, labels)
                buffer.append(accelerator.gather(loss).mean(dim=0).item())
                avg_loss = sum(buffer) / len(buffer)
                pbar.set_postfix_str(f"loss={round(loss.item(), 4)} avg={round(avg_loss, 4)}")
                accelerator.backward(loss)
                # Gradient clipping for stability — capture norm before clipping
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                # Only step scheduler when gradients are actually synchronized
                if accelerator.sync_gradients:
                    scheduler.step()

                # Log to W&B every 20 steps
                if wandb_run and step % 20 == 0:
                    wandb_run.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                        "train/epoch": epoch,
                        "train/step": epoch * len(train_loader) + step,
                    })

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch}: avg_train_loss={round(avg_loss, 4)} ({epoch_time:.0f}s)")

    @torch.inference_mode()
    def validate(epoch: int):
        model.eval()
        preds, labels = [], []
        losses = []
        from tqdm import tqdm
        from collections import defaultdict

        # Per-artist tracking
        per_artist_correct = defaultdict(int)
        per_artist_total = defaultdict(int)
        per_artist_loss = defaultdict(list)
        id_to_artist = {v: k for k, v in artist_to_id.items()}

        val_loss_fn = nn.CrossEntropyLoss(reduction="none")
        for batch in tqdm(val_loader, leave=False, desc=f"Validation {epoch}"):
            seqs, pos, tag = batch
            logits = model(seqs)
            logits_at_eos = logits[torch.arange(logits.shape[0], device=logits.device), pos]
            per_sample_loss = val_loss_fn(logits_at_eos, tag)
            batch_preds = logits_at_eos.argmax(dim=-1)

            for i in range(len(tag)):
                sample_loss = per_sample_loss[i].item()
                losses.append(sample_loss)
                pred = batch_preds[i].item()
                preds.append(pred)
                label = tag[i].item()
                labels.append(label)

                # Per-artist accumulation
                artist = id_to_artist[label]
                per_artist_total[artist] += 1
                per_artist_loss[artist].append(sample_loss)
                if pred == label:
                    per_artist_correct[artist] += 1

        correct = sum(p == t for p, t in zip(preds, labels))
        acc = correct / len(labels) if labels else 0.0
        val_loss = sum(losses) / len(losses) if losses else 0.0

        # Per-artist summary
        logger.info(f"Validation {epoch}: accuracy={round(acc, 4)}, loss={round(val_loss, 4)}")
        logger.info(f"  {'Artist':<20} {'Acc':>8} {'Loss':>8} {'N':>5}")
        for artist in sorted(per_artist_total.keys()):
            a_acc = per_artist_correct[artist] / per_artist_total[artist]
            a_loss = sum(per_artist_loss[artist]) / len(per_artist_loss[artist])
            logger.info(f"  {artist:<20} {a_acc:>7.1%} {a_loss:>8.4f} {per_artist_total[artist]:>5}")

        # Log to W&B
        if wandb_run:
            metrics = {
                "val/accuracy": acc,
                "val/epoch": epoch,
                "val/loss": val_loss,
            }
            # Per-artist metrics
            for artist in per_artist_total:
                a_acc = per_artist_correct[artist] / per_artist_total[artist]
                a_loss = sum(per_artist_loss[artist]) / len(per_artist_loss[artist])
                safe_name = artist.replace(" ", "_")
                metrics[f"val/accuracy_{safe_name}"] = a_acc
                metrics[f"val/loss_{safe_name}"] = a_loss
            wandb_run.log(metrics)

        # Optional confusion matrix
        try:
            if wandb_run:
                import wandb
                class_names = [id_to_artist[i] for i in range(len(artist_to_id))]
                cm = wandb.plot.confusion_matrix(y_true=labels, preds=preds, class_names=class_names)
                wandb_run.log({"val/confusion_matrix": cm})
        except Exception:
            pass

        return acc, val_loss

    metrics = []
    best_acc = 0.0
    best_ckpt_path = os.path.join(project_dir, "checkpoints", "best.pt")
    patience = early_stopping_patience
    stale_epochs = 0
    for epoch in range(num_epochs):
        epoch_wall_start = time.time()
        train_epoch(epoch)
        acc, vloss = validate(epoch)
        epoch_wall_time = time.time() - epoch_wall_start
        metrics.append({"accuracy": acc, "val_loss": vloss})
        if wandb_run:
            wandb_run.log({"epoch/wall_time_sec": epoch_wall_time, "epoch": epoch})

        # Early stopping + best checkpoint
        if acc > best_acc:
            best_acc = acc
            stale_epochs = 0
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), best_ckpt_path)
            logger.info(f"Saved new best checkpoint to {best_ckpt_path}")
        else:
            stale_epochs += 1
            if stale_epochs > patience:
                logger.info("Early stopping: validation accuracy did not improve")
                break

    results = {
        "epoch_metrics": metrics,
        "max_accuracy": max([m["accuracy"] for m in metrics]) if metrics else 0.0,
        "best_checkpoint": best_ckpt_path if os.path.exists(best_ckpt_path) else None,
    }
    with open(os.path.join(project_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Finish W&B run
    if wandb_run:
        wandb_run.finish()
        logger.info("W&B run finished")

    return results


