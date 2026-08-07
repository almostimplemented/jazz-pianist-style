"""Shared utilities for autoregressive generation.

This module consolidates duplicated generation logic from multiple task files:
- Top-k / top-p sampling
- Token-by-token generation loops
- Token format conversion (JSON lists to tuples for vocab lookup)
"""

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample next token from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: Raw logits from model, shape [batch_size, vocab_size]
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top-k tokens (0 = disabled)
        top_p: Keep tokens with cumulative probability <= top_p (1.0 = disabled)

    Returns:
        Sampled token indices, shape [batch_size, 1]
    """
    # Apply temperature
    scaled_logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k_values = torch.topk(scaled_logits, top_k)[0]
        threshold = top_k_values[..., -1, None]
        scaled_logits = torch.where(
            scaled_logits < threshold,
            torch.full_like(scaled_logits, float('-inf')),
            scaled_logits
        )

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        scaled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))

    # Sample from distribution
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
) -> List[int]:
    """Generate tokens autoregressively (single sequence).

    Args:
        model: Language model with forward(input_ids, context?, context_mask?)
        input_ids: Starting token ids, shape [1, seq_len]
        max_new_tokens: Maximum number of new tokens to generate
        eos_id: End-of-sequence token id (stops generation)
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Top-p (nucleus) filtering parameter
        context: Optional cross-attention context, shape [1, context_len, d_model]
        context_mask: Optional context attention mask

    Returns:
        List of generated token ids (excluding prompt)
    """
    generated = []
    device = input_ids.device

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass - handle both base and cross-attention models
            if context is not None:
                logits = model(input_ids, context=context, context_mask=context_mask)
            else:
                logits = model(input_ids)

            # Get logits for last position
            next_logits = logits[:, -1, :]

            # Sample next token
            next_token = sample_next_token(next_logits, temperature, top_k, top_p)

            # Check for EOS
            if next_token.item() == eos_id:
                break

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return generated



def generate_tokens_kv(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
) -> List[int]:
    """Generate tokens using KV-cached inference model.

    Requires model to have been initialized with setup_cache() and to accept
    input_pos parameter (aria.inference.model_cuda API).

    Args:
        model: Inference model with KV cache (setup_cache already called).
        input_ids: Starting token ids, shape [1, seq_len].
        max_new_tokens: Maximum number of new tokens to generate.
        eos_id: End-of-sequence token id.
        temperature: Sampling temperature.
        top_k: Top-k filtering parameter.
        top_p: Top-p (nucleus) filtering parameter.
        context: Optional cross-attention context, shape [1, context_len, d_model].
        context_mask: Optional context attention mask.

    Returns:
        List of generated token ids (excluding prompt).
    """
    generated = []
    device = input_ids.device
    prompt_len = input_ids.shape[1]

    # Reset KV cache before each generation
    if hasattr(model, 'reset_cache'):
        model.reset_cache()

    with torch.no_grad():
        # Prefill: process entire prompt at once
        input_pos = torch.arange(0, prompt_len, device=device)
        if context is not None:
            logits = model(input_ids, input_pos, context=context, context_mask=context_mask)
        else:
            logits = model(input_ids, input_pos)

        # Sample first new token from prompt output
        next_logits = logits[:, -1, :]
        next_token = sample_next_token(next_logits, temperature, top_k, top_p)

        if next_token.item() == eos_id:
            return generated

        generated.append(next_token.item())

        # Decode: one token at a time with KV cache
        for step in range(1, max_new_tokens):
            input_pos = torch.tensor([prompt_len + step - 1], device=device)
            if context is not None:
                logits = model(next_token, input_pos, context=context, context_mask=context_mask)
            else:
                logits = model(next_token, input_pos)

            next_logits = logits[:, -1, :]
            next_token = sample_next_token(next_logits, temperature, top_k, top_p)

            if next_token.item() == eos_id:
                break

            generated.append(next_token.item())

    return generated


def generate_tokens_kv_batched(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """Batched KV-cached generation. Same interface as generate_tokens_kv but
    processes multiple sequences in parallel.

    Args:
        model: Inference model with KV cache. setup_cache must have been called
               with batch_size >= input_ids.shape[0].
        input_ids: [B, prompt_len] — all prompts must be the same length.
        max_new_tokens: Maximum tokens to generate per sequence.
        eos_id: EOS token id.
        temperature, top_k, top_p: Sampling parameters.
        context: Optional [B, context_len, d_model] cross-attention context.
        context_mask: Optional [B, context_len] context mask.

    Returns:
        List of B lists, each containing generated token ids (excluding prompt).
    """
    B = input_ids.shape[0]
    device = input_ids.device
    prompt_len = input_ids.shape[1]

    if hasattr(model, 'reset_cache'):
        model.reset_cache()

    # Pre-allocate output buffer on GPU — no .item() calls in the hot loop
    output = torch.zeros(B, max_new_tokens, dtype=torch.long, device=device)
    lengths = torch.full((B,), max_new_tokens, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)

    with torch.no_grad():
        # Prefill: process entire prompt at once
        input_pos = torch.arange(0, prompt_len, device=device)
        if context is not None:
            logits = model(input_ids, input_pos, context=context, context_mask=context_mask)
        else:
            logits = model(input_ids, input_pos)

        # Sample first new token
        next_tokens = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)  # [B, 1]
        flat = next_tokens.squeeze(-1)  # [B]
        output[:, 0] = flat
        eos_hit = flat == eos_id
        done = done | eos_hit
        lengths = torch.where(eos_hit, torch.zeros_like(lengths), lengths)

        if done.all():
            return [[] for _ in range(B)]

        # Decode: one step at a time with KV cache
        for step in range(1, max_new_tokens):
            input_pos = torch.tensor([prompt_len + step - 1], device=device)
            if context is not None:
                logits = model(next_tokens, input_pos, context=context, context_mask=context_mask)
            else:
                logits = model(next_tokens, input_pos)

            next_tokens = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
            flat = next_tokens.squeeze(-1)
            output[:, step] = flat

            eos_hit = (flat == eos_id) & ~done
            # Record length at first EOS for each sample
            lengths = torch.where(eos_hit, torch.tensor(step, device=device), lengths)
            done = done | eos_hit

            if done.all():
                break

    # Convert to lists once at the end
    generated: List[List[int]] = []
    for b in range(B):
        L = lengths[b].item()
        generated.append(output[b, :L].tolist())
    return generated


def token_to_tuple(tok: Union[list, tuple, str]) -> Union[tuple, str]:
    """Convert JSON-deserialized token to tuple format for vocab lookup.

    JSON deserializes tuples as lists, so we need to convert back.
    Single strings are returned unchanged.

    Args:
        tok: Token as list (from JSON), tuple, or string

    Returns:
        Token as tuple (for compound tokens) or string (for simple tokens)
    """
    return tuple(tok) if isinstance(tok, list) else tok


def tokens_to_ids(
    tokens: List[Union[list, tuple, str]],
    tokenizer,
    skip_unknown: bool = True,
) -> List[int]:
    """Convert a sequence of tokens to vocabulary indices.

    Handles JSON-deserialized tokens (lists) and converts them to tuples
    for vocab lookup.

    Args:
        tokens: List of tokens (as lists, tuples, or strings)
        tokenizer: Tokenizer with .vocab attribute (list of tokens)
        skip_unknown: If True, skip tokens not in vocab. If False, raise error.

    Returns:
        List of token indices
    """
    ids = []
    for tok in tokens:
        tok_tuple = token_to_tuple(tok)
        try:
            ids.append(tokenizer.vocab.index(tok_tuple))
        except ValueError:
            if not skip_unknown:
                raise ValueError(f"Token not in vocab: {tok_tuple}")
            # Skip unknown tokens
            continue
    return ids


def ids_to_tokens(
    ids: List[int],
    tokenizer,
) -> List[Union[tuple, str]]:
    """Convert vocabulary indices back to tokens.

    Args:
        ids: List of token indices
        tokenizer: Tokenizer with .vocab attribute

    Returns:
        List of tokens
    """
    return [tokenizer.vocab[i] for i in ids if i < len(tokenizer.vocab)]
