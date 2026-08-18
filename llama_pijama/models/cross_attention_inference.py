"""Inference-optimized cross-attention model with KV cache.

Extends Aria's inference model (aria.inference.model_cuda) with cross-attention
conditioning while preserving KV cache support for fast autoregressive generation.

Weight-compatible with the training model (cross_attention_model.py) — same
parameter names, just different forward pass mechanics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union

from aria.model import ModelConfig
from aria.inference.model_cuda import (
    KVCache,
    TransformerBlock,
    Transformer as InferenceTransformer,
    TransformerLM as InferenceTransformerLM,
    precompute_freqs_cis,
    apply_rotary_emb,
)

from .cross_attention_model import CrossAttentionLayer


class CrossAttentionInferenceBlock(TransformerBlock):
    """Inference transformer block with optional cross-attention and KV cache.

    Inherits KV-cached self-attention from Aria's TransformerBlock,
    adds gated cross-attention for artist conditioning.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        use_cross_attention: bool = False,
        cross_attention_dropout: float = 0.0,
        cross_attention_gate_init: float = 0.1,
    ):
        super().__init__(model_config)
        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            self.cross_attention = CrossAttentionLayer(
                d_model=model_config.d_model,
                n_heads=model_config.n_heads,
                dropout=cross_attention_dropout,
                gate_init=cross_attention_gate_init,
            )
            self.norm_cross = nn.LayerNorm(model_config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ):
        assert self.kv_cache is not None, "Cache not initialized"

        # Self-attention with KV cache (from parent)
        x = x + self._att_block(
            x=self.norm1(x),
            input_pos=input_pos,
            freqs_cis=freqs_cis,
            mask=mask,
        )

        # Cross-attention (no cache needed — context is fixed)
        if self.use_cross_attention and context is not None:
            cross_out = self.cross_attention(x, context, context_mask)
            x = x + cross_out

        # Feedforward
        x = x + self._ff_block(self.norm2(x))

        return x


class CrossAttentionInferenceTransformer(nn.Module):
    """Inference transformer with cross-attention and KV cache."""

    def __init__(
        self,
        model_config: ModelConfig,
        cross_attention_layers: Optional[List[int]] = None,
        cross_attention_dropout: float = 0.0,
        cross_attention_gate_init: Union[float, Dict[int, float]] = 0.1,
    ):
        super().__init__()
        self.model_config = model_config

        if cross_attention_layers is None:
            cross_attention_layers = list(range(
                max(0, model_config.n_layers - 4), model_config.n_layers
            ))
        self.cross_attention_layers = set(cross_attention_layers)

        self.tok_embeddings = nn.Embedding(model_config.vocab_size, model_config.d_model)

        def get_gate_init(layer_idx: int) -> float:
            if isinstance(cross_attention_gate_init, dict):
                return cross_attention_gate_init.get(layer_idx, 0.1)
            return cross_attention_gate_init

        self.encode_layers = nn.ModuleList([
            CrossAttentionInferenceBlock(
                model_config,
                use_cross_attention=(i in self.cross_attention_layers),
                cross_attention_dropout=cross_attention_dropout,
                cross_attention_gate_init=get_gate_init(i),
            )
            for i in range(model_config.n_layers)
        ])

        self.out_layer_norm = nn.LayerNorm(model_config.d_model)
        self.freqs_cis = None
        self.causal_mask = None

    def forward(
        self,
        idxs: torch.Tensor,
        input_pos: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        pad_idxs: Optional[torch.Tensor] = None,
    ):
        assert self.freqs_cis is not None, "Caches must be initialized first"

        mask = self.causal_mask[input_pos].unsqueeze(0).unsqueeze(0)
        if pad_idxs is not None:
            mask = mask & ~(pad_idxs.unsqueeze(1).unsqueeze(1))

        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idxs)

        for layer in self.encode_layers:
            x = layer(x, input_pos, freqs_cis, mask,
                      context=context, context_mask=context_mask)

        return self.out_layer_norm(x)


class CrossAttentionInferenceLM(nn.Module):
    """Inference LM with cross-attention and KV cache.

    Drop-in replacement for CrossAttentionTransformerLM at inference time.
    Loads the same checkpoint weights — parameter names are compatible.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cross_attention_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model_config = model_config

        if cross_attention_config is None:
            cross_attention_config = {}

        self.model = CrossAttentionInferenceTransformer(
            model_config,
            cross_attention_layers=cross_attention_config.get("layers", None),
            cross_attention_dropout=cross_attention_config.get("dropout", 0.0),
            cross_attention_gate_init=cross_attention_config.get("gate_init", 0.1),
        )

        self.lm_head = nn.Linear(model_config.d_model, model_config.vocab_size, bias=False)

    def forward(
        self,
        idxs: torch.Tensor,
        input_pos: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        pad_idxs: Optional[torch.Tensor] = None,
    ):
        h = self.model(idxs, input_pos, context=context,
                       context_mask=context_mask, pad_idxs=pad_idxs)
        return self.lm_head(h)

    def reset_cache(self):
        """Zero out KV cache buffers for a new generation sequence."""
        for block in self.model.encode_layers:
            if block.kv_cache is not None:
                block.kv_cache.k_cache.zero_()
                block.kv_cache.v_cache.zero_()

    def setup_cache(
        self,
        batch_size: int,
        max_seq_len: int = 4096,
        dtype=torch.bfloat16,
    ):
        head_dim = self.model_config.d_model // self.model_config.n_heads
        # Cache tensors live wherever the model does (CUDA, MPS, or CPU).
        device = next(self.parameters()).device
        for block in self.model.encode_layers:
            block.kv_cache = KVCache(
                max_batch_size=batch_size,
                max_seq_length=max_seq_len,
                n_heads=self.model_config.n_heads,
                head_dim=head_dim,
                dtype=dtype,
            ).to(device)

        self.model.freqs_cis = precompute_freqs_cis(
            seq_len=max_seq_len,
            n_elem=head_dim,
            base=500000,
            dtype=dtype,
        ).to(device)

        self.model.causal_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        ).to(device)
