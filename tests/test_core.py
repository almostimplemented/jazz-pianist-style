"""Core smoke tests: tokenizer vocab contract and model forward/backward.

Run with: uv run pytest tests/ (or python -m pytest).
"""
import torch

from aria.model import ModelConfig

from llama_pijama.tokenization import ConditionalTokenizer
from llama_pijama.models.cross_attention_model import (
    ArtistEmbedding,
    CrossAttentionLayer,
    CrossAttentionTransformerLM,
)


def test_tokenizer_vocab_contract():
    """Base Aria vocab is 17727; PiJAMA-30 extension adds exactly 30 tokens."""
    tok = ConditionalTokenizer()
    assert tok.base_vocab_size == 17727
    assert tok.vocab_size == 17757
    assert tok.num_artist_tokens == 30


def test_tokenizer_explicit_artists_sorted():
    """Token IDs depend on the artist set, not the order it is given in."""
    a = ConditionalTokenizer(artists=["Zoe", "Abe"])
    b = ConditionalTokenizer(artists=["Abe", "Zoe"])
    assert a.artist_tokens["Abe"] == b.artist_tokens["Abe"]
    assert a.vocab_size == 17729


def test_cross_attention_layer_forward():
    layer = CrossAttentionLayer(d_model=64, n_heads=4, use_flash_attention=False)
    out = layer(torch.randn(2, 16, 64), torch.randn(2, 4, 64))
    assert out.shape == (2, 16, 64)


def test_tiny_lm_forward_backward():
    cfg = ModelConfig(
        d_model=64, n_heads=4, n_layers=4, ff_mult=2, drop_p=0.0,
        max_seq_len=128, grad_checkpoint=False, vocab_size=1000,
    )
    lm = CrossAttentionTransformerLM(cfg, {"layers": [2, 3]})
    emb = ArtistEmbedding(num_artists=12, d_model=64)
    ctx, mask = emb(torch.tensor([3, 7]))
    logits = lm(torch.randint(0, 1000, (2, 32)), context=ctx, context_mask=mask)
    assert logits.shape == (2, 32, 1000)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 1000), torch.randint(0, 1000, (64,))
    )
    loss.backward()
    assert torch.isfinite(loss)
