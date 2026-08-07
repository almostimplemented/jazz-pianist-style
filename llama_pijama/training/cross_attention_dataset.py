"""Dataset for cross-attention based conditional generation.

This dataset loads pre-tokenized sequences from JSONL files and provides
artist IDs for cross-attention conditioning. Uses standard Aria tokenization
without vocabulary extension.
"""

import jsonlines
import torch
from pathlib import Path
from typing import Optional, Dict
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class CrossAttentionDataset(Dataset):
    """Dataset for cross-attention conditional music generation.

    Loads pre-tokenized sequences and maps artist names to IDs for
    cross-attention conditioning. No vocabulary extension needed.

    Args:
        jsonl_path: Path to JSONL file with tokenized sequences
        artist_to_id: Mapping from artist names to integer IDs
        max_seq_len: Maximum sequence length
        cache_size: Number of entries to cache in memory
    """

    def __init__(
        self,
        jsonl_path: str,
        artist_to_id: Dict[str, int],
        tokenizer,  # Add tokenizer for encoding sequences
        max_seq_len: int = 4096,
        cache_size: Optional[int] = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.artist_to_id = artist_to_id
        self.tokenizer = tokenizer  # Store tokenizer
        self.max_seq_len = max_seq_len
        self.cache_size = cache_size

        # Validate file exists
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        # Load entries into memory
        self.entries = []
        self.artist_distribution = {}

        with jsonlines.open(self.jsonl_path) as reader:
            for idx, entry in enumerate(reader):
                # Validate entry has required fields
                if "seq" not in entry or "metadata" not in entry:
                    continue
                if "artist" not in entry["metadata"]:
                    continue

                self.entries.append(entry)

                # Track artist distribution
                artist = entry["metadata"]["artist"]
                self.artist_distribution[artist] = self.artist_distribution.get(artist, 0) + 1

                # Apply cache limit if specified
                if cache_size and idx >= cache_size - 1:
                    break

        print(f"Loaded {len(self.entries)} sequences from {self.jsonl_path.name}")
        print(f"Artists: {len(self.artist_distribution)} unique")
        # Dataset introspection logs (diagnostics only)
        try:
            sample_len = min(512, len(self.entries))
            lens = [len(self.entries[i]["seq"]) for i in range(sample_len)]
            if lens:
                lens_sorted = sorted(lens)
                min_len = lens_sorted[0]
                max_len = lens_sorted[-1]
                mean_len = sum(lens) / float(len(lens))
                median_len = lens_sorted[len(lens_sorted) // 2]
                logger.info(f"Seq length stats (first {sample_len}): min={min_len}, median={median_len}, mean={mean_len:.1f}, max={max_len}, max_seq_len={self.max_seq_len}")
            # Top-5 artists by count
            top_artists = sorted(self.artist_distribution.items(), key=lambda kv: kv[1], reverse=True)[:5]
            logger.info(f"Top-5 artists by count: {top_artists}")
            # Tokenizer info
            vocab_size = len(self.tokenizer.vocab)
            unk_tok_id = self.tokenizer.unk_tok_id
            pad_id = self.tokenizer.pad_id
            logger.info(f"Tokenizer info: vocab_size={vocab_size}, unk_tok_id={unk_tok_id}; dataset pad_id={pad_id}")
        except Exception:
            pass

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example.

        Returns:
            Dictionary with:
                - input_ids: Token sequence (standard Aria tokens)
                - labels: Target sequence for loss computation
                - artist_id: Integer ID for cross-attention conditioning
                - metadata: Original metadata from JSONL
        """
        entry = self.entries[idx]
        seq = entry["seq"]
        metadata = entry["metadata"]
        artist = metadata["artist"]

        # Encode the sequence - handle Aria's structured tokens
        encoded_seq = self._encode_sequence(seq)

        # Convert to tensor
        seq = torch.tensor(encoded_seq, dtype=torch.long)

        # Ensure sequence length
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        elif len(seq) < self.max_seq_len:
            # Pad with zeros (will be masked in attention)
            pad_value = int(self.tokenizer.pad_id)
            padding = torch.full((self.max_seq_len - len(seq),), int(pad_value), dtype=torch.long)
            seq = torch.cat([seq, padding])

        # Get artist ID
        artist_id = self.artist_to_id.get(artist, 0)  # Default to 0 if unknown

        # Create input and labels (shifted by one for autoregressive)
        input_ids = seq[:-1]
        labels = seq[1:]
        # Ensure padded positions are ignored by the loss
        # Note: use tokenizer.pad_id if available
        pad_id = self.tokenizer.pad_id
        labels = torch.where(labels == pad_id, torch.tensor(-100, dtype=torch.long), labels)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "artist_id": torch.tensor(artist_id, dtype=torch.long)  # Convert to tensor
            # Don't return metadata - it can't be collated
        }

    def _encode_sequence(self, seq):
        """Encode a sequence of Aria structured tokens to integer IDs.

        Args:
            seq: Sequence of tokens (can be tuples/lists/strings)

        Returns:
            List of integer token IDs
        """
        encoded = []
        for token in seq:
            if isinstance(token, (list, tuple)):
                # This is a structured token like ('prefix', 'instrument', 'piano')
                # Convert list to tuple if needed for vocab lookup
                token_tuple = tuple(token) if isinstance(token, list) else token
                # Find its ID in the vocab
                try:
                    token_id = self.tokenizer.vocab.index(token_tuple)
                except ValueError:
                    # Token not in vocab, use unknown token
                    token_id = self.tokenizer.unk_tok_id
            elif isinstance(token, str):
                # String token
                try:
                    token_id = self.tokenizer.vocab.index(token)
                except ValueError:
                    token_id = self.tokenizer.unk_tok_id
            elif isinstance(token, int):
                # Already an integer ID
                token_id = token
            else:
                # Unknown type, use unknown token
                token_id = self.tokenizer.unk_tok_id

            encoded.append(token_id)

        return encoded


def build_artist_mapping(jsonl_paths):
    """Build artist name to ID mapping from dataset files.

    Args:
        jsonl_paths: List of JSONL file paths to scan for artists

    Returns:
        Dictionary mapping artist names to integer IDs
    """
    all_artists = set()

    for path in jsonl_paths:
        if not Path(path).exists():
            continue

        with jsonlines.open(path) as reader:
            for entry in reader:
                if "metadata" in entry and "artist" in entry["metadata"]:
                    all_artists.add(entry["metadata"]["artist"])

    # Sort for consistency and create mapping
    sorted_artists = sorted(all_artists)
    artist_to_id = {artist: idx for idx, artist in enumerate(sorted_artists)}

    return artist_to_id
