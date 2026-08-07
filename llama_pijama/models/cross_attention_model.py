"""
Cross-Attention Enhanced Aria Model

This module extends the base Aria architecture with cross-attention conditioning
while maintaining compatibility with pretrained weights.

Key features:
- Optional cross-attention layers that can be enabled/disabled
- Preserves all pretrained self-attention weights
- Artist embeddings as conditioning context
- Configurable insertion points (which layers get cross-attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
import math
import logging

from aria.model import (
    FusedEncoderBlock,
    Transformer,
    TransformerLM,
    ModelConfig,
    apply_rotary_emb,
    precompute_freqs_cis
)

logger = logging.getLogger(__name__)

class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for conditioning on artist embeddings.

    This layer performs cross-attention where:
    - Queries come from the main sequence (music tokens)
    - Keys and Values come from the conditioning context (artist embedding)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: Optional[int] = None,
        dropout: float = 0.0,
        gate_init: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head or (d_model // n_heads)
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention

        # Separate projections for Q (from sequence) and K,V (from context)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Layer norm for the query input
        self.norm = nn.LayerNorm(d_model)

        # Optional gating mechanism to control conditioning strength
        # Configurable initialization for experimental control
        self.gate = nn.Parameter(torch.ones(1) * gate_init)

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, d_model) - main sequence
        context: torch.Tensor,  # (batch, context_len, d_model) - artist embeddings
        context_mask: Optional[torch.Tensor] = None  # (batch, seq_len, context_len)
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        context_len = context.shape[1]

        # Normalize input
        x_norm = self.norm(x)

        # Compute queries from sequence
        q = self.q_proj(x_norm)  # (batch, seq_len, d_model)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute keys and values from context
        kv = self.kv_proj(context)  # (batch, context_len, 2 * d_model)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(batch_size, context_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, context_len, self.n_heads, self.d_head).transpose(1, 2)

        # Apply cross-attention
        if self.use_flash_attention and context_mask is None:
            # Use Flash Attention if available
            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False  # Cross-attention is not causal
            )
        else:
            # Manual attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

            if context_mask is not None:
                scores = scores.masked_fill(~context_mask.unsqueeze(1), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        # Apply gating
        return self.gate * output


class CrossAttentionEncoderBlock(FusedEncoderBlock):
    """
    Extended FusedEncoderBlock with optional cross-attention.

    Inherits from the base FusedEncoderBlock to preserve all self-attention
    functionality and weights, while adding cross-attention capability.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        resid_dropout: float = 0.0,
        use_cross_attention: bool = True,
        cross_attention_dropout: float = 0.0,
        cross_attention_gate_init: float = 0.1
    ):
        super().__init__(model_config, resid_dropout)

        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            self.cross_attention = CrossAttentionLayer(
                d_model=model_config.d_model,
                n_heads=model_config.n_heads,
                dropout=cross_attention_dropout,
                gate_init=cross_attention_gate_init
            )
            # Additional layer norm for cross-attention
            self.norm_cross = nn.LayerNorm(model_config.d_model)

        # Instrumentation for debugging attention strengths
        self.record_attention_norms = False
        self.last_self_attn_norm = None
        self.last_cross_attn_norm = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Standard self-attention block (preserves pretrained behavior)
        att_out = self._att_block(self.norm1(x), freqs_cis)

        # Record self-attention residual norm if instrumentation is enabled
        if self.record_attention_norms and self.use_cross_attention:
            with torch.no_grad():
                # Compute L2 norm averaged over batch and sequence dimensions
                self.last_self_attn_norm = torch.norm(att_out, p=2, dim=-1).mean().item()

        x = x + F.dropout(att_out, p=self.resid_dropout, training=self.training)

        # Optional cross-attention block
        if self.use_cross_attention and context is not None:
            cross_out = self.cross_attention(x, context, context_mask)

            # Record cross-attention residual norm if instrumentation is enabled
            if self.record_attention_norms:
                with torch.no_grad():
                    # Compute L2 norm averaged over batch and sequence dimensions
                    self.last_cross_attn_norm = torch.norm(cross_out, p=2, dim=-1).mean().item()

            x = x + F.dropout(cross_out, p=self.resid_dropout, training=self.training)

        # Standard feedforward block
        ff_out = self._ff_block(self.norm2(x))
        x = x + F.dropout(ff_out, p=self.resid_dropout, training=self.training)

        return x


class CrossAttentionTransformer(Transformer):
    """
    Transformer with cross-attention conditioning support.

    Extends the base Transformer to use CrossAttentionEncoderBlocks
    and handle artist conditioning context.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cross_attention_layers: Optional[List[int]] = None,
        cross_attention_dropout: float = 0.0,
        cross_attention_gate_init: Union[float, Dict[int, float]] = 0.1
    ):
        # Don't call parent __init__ to avoid creating standard blocks
        nn.Module.__init__(self)  # Call grandparent init

        self.model_config = model_config
        self.n_layers = model_config.n_layers

        # Determine which layers should have cross-attention
        if cross_attention_layers is None:
            # Default to last 4 layers (top quarter) for 16-layer model
            cross_attention_layers = list(range(max(0, model_config.n_layers - 4), model_config.n_layers))
            logger.warning(f"cross_attention_layers=None interpreted as last 4 layers: {cross_attention_layers}")
        self.cross_attention_layers = set(cross_attention_layers)

        # Token embeddings (preserved from base)
        self.tok_embeddings = nn.Embedding(model_config.vocab_size, model_config.d_model)

        # Helper to get gate init for a specific layer
        def get_gate_init(layer_idx: int) -> float:
            if isinstance(cross_attention_gate_init, dict):
                return cross_attention_gate_init.get(layer_idx, 0.1)  # Default 0.1 if not specified
            return cross_attention_gate_init

        # Log layer-wise gate configuration
        if isinstance(cross_attention_gate_init, dict):
            logger.info(f"Using layer-wise gate initialization: {cross_attention_gate_init}")

        # Create encoder blocks with selective cross-attention
        self.encode_layers = nn.ModuleList([
            CrossAttentionEncoderBlock(
                model_config,
                resid_dropout=model_config.resid_dropout,
                use_cross_attention=(i in self.cross_attention_layers),
                cross_attention_dropout=cross_attention_dropout,
                cross_attention_gate_init=get_gate_init(i)
            )
            for i in range(model_config.n_layers)
        ])

        # Final layer norm (preserved from base)
        self.out_layer_norm = nn.LayerNorm(model_config.d_model)

        # Precompute rotary embeddings using Aria's function
        self.freqs_cis = None  # Will be computed on first forward pass like Aria does

    def set_attention_instrumentation(self, enabled: bool = True):
        """Enable or disable attention norm recording for all layers."""
        for layer in self.encode_layers:
            if isinstance(layer, CrossAttentionEncoderBlock):
                layer.record_attention_norms = enabled

    def get_attention_stats(self) -> List[Dict[str, float]]:
        """Collect attention statistics from all layers with cross-attention.

        Returns:
            List of dicts with layer stats including:
            - layer_idx: Layer index
            - self_attn_norm: L2 norm of self-attention residual
            - cross_attn_norm: L2 norm of cross-attention residual
            - ratio: cross_attn_norm / self_attn_norm
            - gate: Current gate parameter value
        """
        stats = []
        for idx, layer in enumerate(self.encode_layers):
            if isinstance(layer, CrossAttentionEncoderBlock) and layer.use_cross_attention:
                if layer.last_self_attn_norm is not None and layer.last_cross_attn_norm is not None:
                    # Get gate value if it exists
                    gate_value = 1.0
                    if hasattr(layer, 'cross_attention') and hasattr(layer.cross_attention, 'gate'):
                        gate_value = layer.cross_attention.gate.item()

                    ratio = layer.last_cross_attn_norm / (layer.last_self_attn_norm + 1e-8)

                    stats.append({
                        'layer_idx': idx,
                        'self_attn_norm': layer.last_self_attn_norm,
                        'cross_attn_norm': layer.last_cross_attn_norm,
                        'ratio': ratio,
                        'gate': gate_value
                    })

                    # Reset after collecting
                    layer.last_self_attn_norm = None
                    layer.last_cross_attn_norm = None

        return stats

    def forward(
        self,
        src: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None  # For compatibility with base
    ) -> torch.Tensor:
        batch_size, seq_len = src.shape

        # Token embeddings
        h = self.tok_embeddings(src)

        # Handle optional conditioning embedding (from base model)
        if cond is not None:
            h = torch.cat([cond.unsqueeze(1), h[:, :-1]], dim=1)

        # Compute frequency embeddings if not already done (following Aria's approach)
        head_dim = self.model_config.d_model // self.model_config.n_heads
        if self.freqs_cis is None or self.freqs_cis.size(0) < seq_len:
            # Precompute only up to the needed length (grow on demand)
            target_len = seq_len
            self.freqs_cis = precompute_freqs_cis(
                seq_len=target_len,
                n_elem=head_dim,
                base=500000,
            ).to(src.device)
            try:
                logger.info(f"precompute_freqs_cis: cached_len={target_len}, head_dim={head_dim}")
            except Exception:
                pass
        freqs_cis = self.freqs_cis[:seq_len]

        # Apply encoder layers with optional cross-attention
        for layer in self.encode_layers:
            if isinstance(layer, CrossAttentionEncoderBlock) and layer.use_cross_attention:
                h = layer(h, freqs_cis, context=context, context_mask=context_mask)
            else:
                h = layer(h, freqs_cis)

        # Final norm
        return self.out_layer_norm(h)


class CrossAttentionTransformerLM(TransformerLM):
    """
    Language model with cross-attention conditioning.

    Extends TransformerLM to use CrossAttentionTransformer.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cross_attention_config: Optional[dict] = None
    ):
        # Initialize as parent but replace the transformer
        nn.Module.__init__(self)  # Skip parent init

        self.model_config = model_config

        # Parse cross-attention configuration
        if cross_attention_config is None:
            cross_attention_config = {}

        # Create cross-attention transformer
        self.model = CrossAttentionTransformer(
            model_config,
            cross_attention_layers=cross_attention_config.get('layers', None),
            cross_attention_dropout=cross_attention_config.get('dropout', 0.0),
            cross_attention_gate_init=cross_attention_config.get('gate_init', 0.1)  # Can be float or dict
        )

        # Language model head (preserved from base)
        self.lm_head = nn.Linear(model_config.d_model, model_config.vocab_size, bias=False)

    def forward(
        self,
        src: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.model(src, context=context, context_mask=context_mask, cond=cond)
        return self.lm_head(h)

    def load_pretrained_weights(self, state_dict: dict, strict: bool = False):
        """
        Load pretrained Aria weights into the cross-attention model.

        This method handles the mapping from standard FusedEncoderBlock
        weights to CrossAttentionEncoderBlock weights.
        """
        # Create a new state dict with mapped keys
        new_state_dict = {}

        # Log embedding key and size if present
        try:
            embed_keys = [k for k in state_dict.keys() if k.endswith("tok_embeddings.weight")]
            if embed_keys:
                logger.info(f"Pretrained embedding key: {embed_keys[0]}, size={tuple(state_dict[embed_keys[0]].shape)}")
        except Exception:
            pass

        for key, value in state_dict.items():
            # Most keys can be loaded directly
            new_state_dict[key] = value

            # The encoder blocks will have the same parameter names for
            # self-attention components, so they should load directly

        # Load with strict=False to ignore missing cross-attention weights
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)

        # Report what happened
        cross_attn_missing = [k for k in missing_keys if 'cross_attention' in k or 'norm_cross' in k]
        other_missing = [k for k in missing_keys if k not in cross_attn_missing]

        if other_missing and strict:
            raise RuntimeError(f"Missing required keys: {other_missing}")

        try:
            logger.info(f"Loaded pretrained weights successfully.")
            logger.info(f"Initialized {len(cross_attn_missing)} new cross-attention parameters.")
            if other_missing:
                preview = other_missing[:20]
                logger.warning(f"{len(other_missing)} non-cross-attn parameters missing during load (showing up to 20): {preview}")
            if unexpected_keys:
                upreview = unexpected_keys[:10]
                logger.warning(f"{len(unexpected_keys)} unexpected keys in pretrained state (showing up to 10): {upreview}")
        except Exception:
            pass

        return missing_keys, unexpected_keys


class ArtistEmbedding(nn.Module):
    """
    Learnable artist embeddings for conditioning.

    Creates a set of learnable embeddings for each artist that can be used
    as context in cross-attention layers.
    """

    def __init__(
        self,
        num_artists: int,
        d_model: int,
        context_length: int = 4,  # Number of context vectors per artist
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_artists = num_artists
        self.d_model = d_model
        self.context_length = context_length

        # Learnable embeddings for each artist
        # Shape: (num_artists, context_length, d_model)
        self.embeddings = nn.Parameter(torch.randn(num_artists, context_length, d_model) * 0.02)

        # Optional projection layer to transform embeddings
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, artist_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get artist embeddings for a batch of artist IDs.

        Args:
            artist_ids: (batch_size,) tensor of artist IDs

        Returns:
            context: (batch_size, context_length, d_model) tensor of context vectors
            mask: (batch_size, 1, context_length) attention mask (all ones for now)
        """
        batch_size = artist_ids.shape[0]

        # Gather embeddings for the specified artists
        context = self.embeddings[artist_ids]  # (batch, context_length, d_model)

        # Apply projection and normalization
        context = self.projection(context)
        context = self.norm(context)
        context = self.dropout(context)

        # Create attention mask (all positions are valid for now)
        mask = torch.ones(batch_size, 1, self.context_length, device=context.device, dtype=torch.bool)

        return context, mask

    def interpolate(self, artist_ids: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between multiple artists for style mixing.

        Args:
            artist_ids: (batch_size, num_artists) tensor of artist IDs
            weights: (batch_size, num_artists) tensor of interpolation weights

        Returns:
            context: (batch_size, context_length, d_model) interpolated context
            mask: (batch_size, 1, context_length) attention mask
        """
        batch_size, num_artists = artist_ids.shape

        # Gather embeddings for all artists
        all_contexts = self.embeddings[artist_ids]  # (batch, num_artists, context_length, d_model)

        # Apply weights for interpolation
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (batch, num_artists, 1, 1)
        context = (all_contexts * weights).sum(dim=1)  # (batch, context_length, d_model)

        # Apply projection and normalization
        context = self.projection(context)
        context = self.norm(context)
        context = self.dropout(context)

        # Create attention mask
        mask = torch.ones(batch_size, 1, self.context_length, device=context.device, dtype=torch.bool)

        return context, mask


# Utility function to convert standard Aria model to cross-attention version
def convert_aria_to_cross_attention(
    aria_model: TransformerLM,
    cross_attention_config: Optional[dict] = None,
    num_artists: int = 30
) -> Tuple[CrossAttentionTransformerLM, ArtistEmbedding]:
    """
    Convert a pretrained Aria model to a cross-attention version.

    Args:
        aria_model: Pretrained Aria TransformerLM
        cross_attention_config: Configuration for cross-attention layers
        num_artists: Number of artists for embedding table

    Returns:
        cross_model: CrossAttentionTransformerLM with loaded weights
        artist_embeddings: ArtistEmbedding module
    """
    # Get model configuration
    model_config = aria_model.model_config

    # Create cross-attention model
    cross_model = CrossAttentionTransformerLM(model_config, cross_attention_config)

    # Load pretrained weights
    cross_model.load_pretrained_weights(aria_model.state_dict())

    # Create artist embeddings
    artist_embeddings = ArtistEmbedding(
        num_artists=num_artists,
        d_model=model_config.d_model,
        context_length=cross_attention_config.get('context_length', 4),
        dropout=cross_attention_config.get('embedding_dropout', 0.0)
    )

    return cross_model, artist_embeddings