"""Conditioned generation smoke: real checkpoint emits a valid event stream.

Skipped unless the generator checkpoint is present locally.
"""
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "generator_best"

pytestmark = pytest.mark.skipif(
    not (CKPT / "model.safetensors").exists(),
    reason="generator checkpoint not downloaded",
)


def test_conditioned_generation_produces_notes():
    from safetensors.torch import load_file
    from aria.config import load_model_config
    from aria.model import ModelConfig
    from ariautils.tokenizer import AbsTokenizer
    from llama_pijama.models import ArtistEmbedding, CrossAttentionTransformerLM
    from llama_pijama.utils.generation import generate_tokens

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = load_model_config("medium")
    cfg.setdefault("resid_dropout", 0.0)
    cfg["grad_checkpoint"] = False
    model = CrossAttentionTransformerLM(
        ModelConfig(**cfg),
        {"layers": [8, 9, 10, 11, 12, 13, 14, 15], "dropout": 0.1, "gate_init": 0.1},
    )
    model.load_state_dict(load_file(CKPT / "model.safetensors"))
    emb = ArtistEmbedding(num_artists=12, d_model=cfg["d_model"], context_length=4)
    emb.load_state_dict(load_file(CKPT / "artist_embeddings.safetensors"))
    model = model.to(device).eval()
    emb = emb.to(device).eval()

    artist_to_id = json.loads((ROOT / "data/eval/artist_to_id.json").read_text())
    tok = AbsTokenizer()
    ctx, mask = emb(torch.tensor([artist_to_id["Art Tatum"]], device=device))
    prefix = tok.vocab.index(("prefix", "instrument", "piano"))
    eos = tok.vocab.index(tok.eos_tok)
    with torch.no_grad():
        out = generate_tokens(model, torch.tensor([[prefix]], device=device),
                              64, eos, temperature=0.95, context=ctx, context_mask=mask)
    events = [tok.vocab[t] for t in out if isinstance(t, int) and t < len(tok.vocab)]
    notes = [e for e in events if isinstance(e, tuple) and e and e[0] == "piano"]
    assert len(notes) >= 5, f"only {len(notes)} note events in 64 tokens"
