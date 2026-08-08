"""Checkpoint parity: released weights load into these classes with zero drift.

Skipped unless the generator checkpoint is present locally (see README for
download instructions).
"""
from pathlib import Path

import pytest
from safetensors.torch import load_file

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "generator_best"

pytestmark = pytest.mark.skipif(
    not (CKPT / "model.safetensors").exists(),
    reason="generator checkpoint not downloaded",
)


def test_generator_state_dict_parity():
    from aria.config import load_model_config
    from aria.model import ModelConfig
    from llama_pijama.models import ArtistEmbedding, CrossAttentionTransformerLM

    cfg = load_model_config("medium")
    cfg.setdefault("resid_dropout", 0.0)
    model = CrossAttentionTransformerLM(
        ModelConfig(**cfg),
        {"layers": [8, 9, 10, 11, 12, 13, 14, 15], "dropout": 0.1, "gate_init": 0.1},
    )
    sd = load_file(CKPT / "model.safetensors")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected

    emb = ArtistEmbedding(num_artists=12, d_model=cfg["d_model"], context_length=4)
    esd = load_file(CKPT / "artist_embeddings.safetensors")
    m, u = emb.load_state_dict(esd, strict=False)
    assert not m and not u

    # Trained gate values, not bare initialization
    gates = [
        float(layer.cross_attention.gate.detach())
        for layer in model.model.encode_layers
        if getattr(layer, "use_cross_attention", False)
    ]
    assert len(gates) == 8
    assert any(abs(g - 0.1) > 1e-4 for g in gates)
