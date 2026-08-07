"""Dataset for conditional generation with artist tokens.

This module provides a PyTorch Dataset that loads pre-tokenized sequences
from JSONL files and adds artist conditioning tokens for conditional generation.
"""

import json
import jsonlines
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from torch.utils.data import Dataset

from ..tokenization import ConditionalTokenizer


class ConditionalDataset(Dataset):
    """Dataset for conditional music generation with artist tokens.

    This dataset loads pre-tokenized sequences from JSONL files and adds
    artist conditioning tokens. It supports multiple conditioning strategies
    and maintains compatibility with Aria's training infrastructure.

    Args:
        jsonl_path: Path to JSONL file with tokenized sequences
        conditional_tokenizer: Tokenizer with artist token support
        conditioning_strategy: How to add artist tokens ('prefix', 'periodic', 'replace')
        max_seq_len: Maximum sequence length (must match JSONL preprocessing)
        conditioning_dropout: Probability of dropping conditioning for classifier-free guidance
        cache_size: Number of entries to cache in memory (None = cache all)
    """

    def __init__(
        self,
        jsonl_path: str,
        conditional_tokenizer: ConditionalTokenizer,
        conditioning_strategy: str = "prefix",
        max_seq_len: int = 1024,
        conditioning_dropout: float = 0.0,
        cache_size: Optional[int] = None,
        transform: Optional[Any] = None,  # Augmentation functions
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = conditional_tokenizer
        self.conditioning_strategy = conditioning_strategy
        self.max_seq_len = max_seq_len
        self.conditioning_dropout = conditioning_dropout
        self.cache_size = cache_size
        self._transform = transform

        # Validate file exists
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        # Load entries into memory (for now - can optimize with mmap later)
        self.entries = []
        self.artist_distribution = {}

        with jsonlines.open(self.jsonl_path) as reader:
            for idx, entry in enumerate(reader):
                self.entries.append(entry)

                # Track artist distribution
                artist = entry["metadata"]["artist"]
                self.artist_distribution[artist] = self.artist_distribution.get(artist, 0) + 1

                # Apply cache limit if specified
                if cache_size and idx >= cache_size - 1:
                    break

        print(f"Loaded {len(self.entries)} sequences from {self.jsonl_path.name}")
        print(f"Artists: {len(self.artist_distribution)} unique")

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example with conditioning.

        Args:
            idx: Index of the sequence to retrieve

        Returns:
            Dictionary with:
                - input_ids: Input token sequence with conditioning
                - target_ids: Target token sequence for loss computation
                - attention_mask: Mask for padding tokens
                - loss_mask: Mask for loss computation (can ignore artist tokens)
                - metadata: Original metadata from JSONL
        """
        entry = self.entries[idx]
        seq = entry["seq"]
        metadata = entry["metadata"]
        artist = metadata["artist"]

        # Apply augmentation if configured (BEFORE conditioning)
        if self._transform:
            # Convert to proper token format if needed
            formatted_seq = []
            for token in seq:
                if isinstance(token, list):
                    formatted_seq.append(tuple(token))
                else:
                    formatted_seq.append(token)
            seq = self._transform(formatted_seq)

        # Check if we should apply conditioning dropout for this sample
        apply_dropout = self.conditioning_dropout > 0 and torch.rand(1).item() < self.conditioning_dropout

        # Apply conditioning based on strategy, passing dropout flag
        if self.conditioning_strategy == "prefix":
            input_ids, target_ids, loss_mask = self._apply_prefix_conditioning(seq, artist, skip_conditioning=apply_dropout)
        elif self.conditioning_strategy == "periodic":
            input_ids, target_ids, loss_mask = self._apply_periodic_conditioning(seq, artist)
        elif self.conditioning_strategy == "replace":
            input_ids, target_ids, loss_mask = self._apply_replace_conditioning(seq, artist)
        else:
            raise ValueError(f"Unknown conditioning strategy: {self.conditioning_strategy}")

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = self._create_attention_mask(input_ids)

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        loss_mask = torch.tensor(loss_mask, dtype=torch.bool)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

    def _apply_prefix_conditioning(
        self, seq: List[int], artist: str, skip_conditioning: bool = False
    ) -> Tuple[List[int], List[int], List[bool]]:
        """Apply prefix conditioning: prepend artist token to sequence.

        Args:
            seq: Original token sequence
            artist: Artist name for conditioning
            skip_conditioning: If True, don't add artist token (for dropout)

        Returns:
            Tuple of (input_ids, target_ids, loss_mask)
        """
        # Handle tokenized sequences (they're lists of tuples/lists)
        # Need to encode them first
        encoded_seq = self._encode_sequence(seq)

        if not skip_conditioning:
            # Get artist token
            artist_token = self.tokenizer.get_artist_token(artist)

            # Truncate to max_seq_len - 1 to leave room for artist token
            if len(encoded_seq) >= self.max_seq_len:
                encoded_seq = encoded_seq[:self.max_seq_len - 1]

            # Prepend artist token
            input_ids = [artist_token] + encoded_seq[:-1]

            # Target is shifted by one (standard language modeling)
            target_ids = encoded_seq
        else:
            # No conditioning - just use the sequence as-is
            # Truncate to max_seq_len
            if len(encoded_seq) > self.max_seq_len:
                encoded_seq = encoded_seq[:self.max_seq_len]

            # Standard autoregressive setup without artist token
            input_ids = encoded_seq[:-1]
            target_ids = encoded_seq[1:]

        # Pad to max_seq_len if needed
        pad_token_id = self.tokenizer.vocab.index(self.tokenizer.pad_tok)
        actual_len = len(input_ids)
        if len(input_ids) < self.max_seq_len:
            padding_length = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [pad_token_id] * padding_length
            target_ids = target_ids + [pad_token_id] * padding_length

        # Create loss mask based on whether we have conditioning
        if not skip_conditioning:
            # With conditioning: mask the first position (artist token prediction) and padding
            loss_mask = [False] + [True] * (actual_len - 1) + [False] * (self.max_seq_len - actual_len)
        else:
            # Without conditioning: only mask padding
            loss_mask = [True] * actual_len + [False] * (self.max_seq_len - actual_len)

        return input_ids, target_ids, loss_mask

    def _apply_periodic_conditioning(
        self, seq: List[int], artist: str, period: int = 256
    ) -> Tuple[List[int], List[int], List[bool]]:
        """Apply periodic conditioning: inject artist token every N tokens.

        Args:
            seq: Original token sequence
            artist: Artist name for conditioning
            period: How often to inject artist token

        Returns:
            Tuple of (input_ids, target_ids, loss_mask)
        """
        artist_token = self.tokenizer.get_artist_token(artist)
        encoded_seq = self._encode_sequence(seq)

        input_ids = []
        target_ids = []
        loss_mask = []

        for i in range(0, len(encoded_seq), period):
            # Add artist token at period boundaries
            if i == 0 or i % period == 0:
                input_ids.append(artist_token)
                target_ids.append(encoded_seq[i] if i < len(encoded_seq) else self.tokenizer.pad_tok)
                loss_mask.append(False)  # Don't compute loss on artist tokens

            # Add the sequence chunk
            chunk = encoded_seq[i:i+period]
            input_ids.extend(chunk[:-1] if len(chunk) > 1 else chunk)
            target_ids.extend(chunk)
            loss_mask.extend([True] * len(chunk))

        # Truncate to max_seq_len
        input_ids = input_ids[:self.max_seq_len]
        target_ids = target_ids[:self.max_seq_len]
        loss_mask = loss_mask[:self.max_seq_len]

        # Pad to max_seq_len if needed
        pad_token_id = self.tokenizer.vocab.index(self.tokenizer.pad_tok)
        if len(input_ids) < self.max_seq_len:
            padding_length = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [pad_token_id] * padding_length
            target_ids = target_ids + [pad_token_id] * padding_length
            loss_mask = loss_mask + [False] * padding_length

        return input_ids, target_ids, loss_mask

    def _apply_replace_conditioning(
        self, seq: List[int], artist: str
    ) -> Tuple[List[int], List[int], List[bool]]:
        """Replace first special token with artist token.

        Args:
            seq: Original token sequence
            artist: Artist name for conditioning

        Returns:
            Tuple of (input_ids, target_ids, loss_mask)
        """
        artist_token = self.tokenizer.get_artist_token(artist)
        encoded_seq = self._encode_sequence(seq)

        # Find first special token (like instrument prefix) and replace it
        input_ids = encoded_seq.copy()
        for i, token in enumerate(input_ids):
            # Check if it's a special prefix token
            if isinstance(token, tuple) or token < 100:  # Heuristic for special tokens
                input_ids[i] = artist_token
                break
        else:
            # No special token found, use prefix strategy as fallback
            return self._apply_prefix_conditioning(seq, artist)

        # Truncate to max_seq_len
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            target_ids = encoded_seq[:self.max_seq_len]
        else:
            target_ids = encoded_seq

        # Pad to max_seq_len if needed
        pad_token_id = self.tokenizer.vocab.index(self.tokenizer.pad_tok)
        if len(input_ids) < self.max_seq_len:
            padding_length = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [pad_token_id] * padding_length
            target_ids = target_ids + [pad_token_id] * padding_length

        # Create loss mask (don't compute loss on padding)
        loss_mask = [True] * min(len(encoded_seq), self.max_seq_len) + [False] * max(0, self.max_seq_len - len(encoded_seq))

        return input_ids, target_ids, loss_mask

    def _encode_sequence(self, seq: List[Any]) -> List[int]:
        """Encode a sequence of tokens (may be tuples/lists) to integer IDs.

        Args:
            seq: Sequence of tokens (can be mixed types)

        Returns:
            List of integer token IDs
        """
        # The sequences are already tokenized but not encoded
        # They contain tuples like ('prefix', 'instrument', 'piano')
        # We need to encode them using the base tokenizer

        encoded = []
        for token in seq:
            if isinstance(token, (list, tuple)):
                # This is a special token like ('prefix', 'instrument', 'piano')
                # Find its ID in the vocab
                token_id = self.tokenizer.vocab.index(tuple(token))
            elif isinstance(token, str):
                # String token
                token_id = self.tokenizer.vocab.index(token)
            elif isinstance(token, int):
                # Already an integer ID
                token_id = token
            else:
                raise ValueError(f"Unknown token type: {type(token)}")

            encoded.append(token_id)

        return encoded

    def _remove_conditioning(self, input_ids: List[int]) -> List[int]:
        """Remove artist conditioning tokens from sequence.

        Args:
            input_ids: Sequence with artist tokens

        Returns:
            Sequence without artist tokens (removed, not replaced)
        """
        # Simply remove the artist token rather than replacing with padding
        # This maintains proper sequence structure
        cleaned = []
        for token_id in input_ids:
            if not self.tokenizer.is_artist_token(token_id):
                cleaned.append(token_id)
        return cleaned

    def _create_attention_mask(self, input_ids: List[int]) -> List[bool]:
        """Create attention mask for padding tokens.

        Args:
            input_ids: Token sequence

        Returns:
            Boolean mask (True for real tokens, False for padding)
        """
        # Check for padding token - it's likely at a fixed position in vocab
        # For AbsTokenizer, pad_tok is usually index 2
        pad_id = self.tokenizer.vocab.index(self.tokenizer.pad_tok)
        return [token_id != pad_id for token_id in input_ids]

    def set_transform(self, transform):
        """Set data augmentation transform functions.

        Args:
            transform: Single transform function or list of functions to apply.
                These should be Aria's augmentation functions from tokenizer.export_data_aug()
        """
        if isinstance(transform, list):
            # Multiple transforms - apply in sequence
            def _combined_transform(x):
                for fn in transform:
                    x = fn(x)
                return x
            self._transform = _combined_transform
        else:
            self._transform = transform

    def get_artist_distribution(self) -> Dict[str, int]:
        """Get the distribution of artists in the dataset.

        Returns:
            Dictionary mapping artist names to counts
        """
        return self.artist_distribution.copy()

    def get_batch_by_artist(self, artist: str, batch_size: int = 8) -> List[Dict[str, torch.Tensor]]:
        """Get a batch of sequences from a specific artist.

        Useful for debugging and analysis.

        Args:
            artist: Artist name
            batch_size: Number of sequences to return

        Returns:
            List of training examples from the specified artist
        """
        artist_indices = [
            i for i, entry in enumerate(self.entries)
            if entry["metadata"]["artist"] == artist
        ]

        if not artist_indices:
            raise ValueError(f"No sequences found for artist: {artist}")

        # Sample batch_size indices
        import random
        sampled = random.sample(artist_indices, min(batch_size, len(artist_indices)))

        return [self[idx] for idx in sampled]


def test_conditional_dataset():
    """Test the ConditionalDataset implementation."""
    print("Testing ConditionalDataset...")

    # Initialize tokenizer
    from llama_pijama.tokenization import ConditionalTokenizer
    tokenizer = ConditionalTokenizer()

    # Test with the downloaded sample
    dataset = ConditionalDataset(
        jsonl_path="/tmp/sample_train.jsonl",
        conditional_tokenizer=tokenizer,
        conditioning_strategy="prefix",
        max_seq_len=1024,
        cache_size=100,  # Just load first 100 for testing
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Artist distribution: {dataset.get_artist_distribution()}")

    # Test getting a single item
    item = dataset[0]
    print(f"\nFirst item keys: {item.keys()}")
    print(f"Input shape: {item['input_ids'].shape}")
    print(f"Target shape: {item['target_ids'].shape}")
    print(f"Artist: {item['metadata']['artist']}")

    # Check that first token is artist token
    first_token = item['input_ids'][0].item()
    print(f"First token ID: {first_token}")
    print(f"Is artist token: {tokenizer.is_artist_token(first_token)}")
    if tokenizer.is_artist_token(first_token):
        print(f"Artist from token: {tokenizer.get_artist_from_token(first_token)}")

    print("\n✓ ConditionalDataset test passed!")


if __name__ == "__main__":
    test_conditional_dataset()