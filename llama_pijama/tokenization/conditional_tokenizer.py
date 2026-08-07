"""Conditional tokenizer extending AbsTokenizer with artist tokens."""

import os
import json
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from ariautils.tokenizer import AbsTokenizer


class ConditionalTokenizer(AbsTokenizer):
    """Tokenizer extended with artist conditioning tokens.

    Adds special tokens for each artist in PiJAMA-30, allowing for
    conditional generation by prefixing sequences with artist tokens.
    """

    DEFAULT_ARTISTS_CSV = Path(__file__).resolve().parents[2] / "data" / "pijama_30.csv"

    def __init__(
        self,
        artists: Optional[List[str]] = None,
        pijama_csv_path: Optional[str] = None,
    ):
        """Initialize conditional tokenizer with artist tokens.

        Args:
            artists: Explicit artist names. Sorted internally, so token IDs
                depend only on the set, not the given order.
            pijama_csv_path: CSV with an ``artist`` column; used when
                ``artists`` is None. Defaults to the bundled
                ``data/pijama_30.csv``.
        """
        super().__init__()

        # Store the base vocabulary size before adding artist tokens
        self.base_vocab_size = self.vocab_size

        if artists is None:
            csv_path = Path(pijama_csv_path or self.DEFAULT_ARTISTS_CSV)
            if not csv_path.exists():
                raise FileNotFoundError(f"Artist CSV not found at {csv_path}")
            artists = pd.read_csv(csv_path)["artist"].unique()

        artists = sorted(artists)

        # Create artist token mappings
        # Artist tokens start immediately after the base vocabulary
        self.artist_tokens = {}
        self.artist_token_to_name = {}

        for idx, artist in enumerate(artists):
            token_id = self.base_vocab_size + idx
            # Use underscore to replace spaces for cleaner token names
            token_name = f"<ARTIST_{artist.replace(' ', '_')}>"

            self.artist_tokens[artist] = token_id
            self.artist_token_to_name[token_id] = artist

            # Also create a mapping from token name to ID for convenience
            self.artist_tokens[token_name] = token_id

        # Update total vocabulary size
        self.vocab_size = self.base_vocab_size + len(artists)
        self.num_artist_tokens = len(artists)

        # Store special token ranges for easy checking
        self.artist_token_range = (self.base_vocab_size, self.vocab_size)

    def get_artist_token(self, artist: str) -> int:
        """Get token ID for a given artist.

        Args:
            artist: Artist name

        Returns:
            Token ID for the artist

        Raises:
            KeyError: If artist not in vocabulary
        """
        if artist not in self.artist_tokens:
            raise KeyError(f"Artist '{artist}' not in vocabulary. "
                         f"Available artists: {sorted([a for a in self.artist_tokens.keys() if not a.startswith('<')])}")
        return self.artist_tokens[artist]

    def is_artist_token(self, token_id: int) -> bool:
        """Check if a token ID corresponds to an artist token.

        Args:
            token_id: Token ID to check

        Returns:
            True if token is an artist token, False otherwise
        """
        return self.artist_token_range[0] <= token_id < self.artist_token_range[1]

    def get_artist_from_token(self, token_id: int) -> Optional[str]:
        """Get artist name from token ID.

        Args:
            token_id: Token ID

        Returns:
            Artist name if token is an artist token, None otherwise
        """
        return self.artist_token_to_name.get(token_id)

    def add_conditioning(self, seq: List[int], artist: str) -> List[int]:
        """Add artist conditioning token to the beginning of a sequence.

        Args:
            seq: Token sequence
            artist: Artist name for conditioning

        Returns:
            Conditioned sequence with artist token prepended
        """
        artist_token = self.get_artist_token(artist)
        return [artist_token] + seq

    def remove_conditioning(self, seq: List[int]) -> List[int]:
        """Remove artist conditioning token from sequence if present.

        Args:
            seq: Token sequence potentially with artist token

        Returns:
            Sequence without artist token
        """
        if len(seq) > 0 and self.is_artist_token(seq[0]):
            return seq[1:]
        return seq

    def decode_with_artist(self, seq: List[int]) -> Dict[str, any]:
        """Decode sequence and extract artist information.

        Args:
            seq: Token sequence

        Returns:
            Dictionary with 'artist' and 'music' fields
        """
        artist = None
        music_seq = seq

        if len(seq) > 0 and self.is_artist_token(seq[0]):
            artist = self.get_artist_from_token(seq[0])
            music_seq = seq[1:]

        return {
            'artist': artist,
            'music': self.decode(music_seq) if hasattr(self, 'decode') else music_seq
        }

    def get_vocab_info(self) -> Dict[str, any]:
        """Get information about the vocabulary.

        Returns:
            Dictionary with vocabulary statistics
        """
        return {
            'total_vocab_size': self.vocab_size,
            'base_vocab_size': self.base_vocab_size,
            'num_artist_tokens': self.num_artist_tokens,
            'artist_token_range': self.artist_token_range,
            'artists': sorted([a for a in self.artist_tokens.keys() if not a.startswith('<')])
        }


def test_conditional_tokenizer():
    """Test the ConditionalTokenizer implementation."""

    # Initialize tokenizer
    tokenizer = ConditionalTokenizer()

    # Print vocabulary info
    info = tokenizer.get_vocab_info()
    print(f"Total vocab size: {info['total_vocab_size']}")
    print(f"Base vocab size: {info['base_vocab_size']}")
    print(f"Number of artist tokens: {info['num_artist_tokens']}")
    print(f"Artist token range: {info['artist_token_range']}")
    print(f"First 5 artists: {info['artists'][:5]}")

    # Test artist token retrieval
    artist = "Bill Cunliffe"
    token_id = tokenizer.get_artist_token(artist)
    print(f"\nToken ID for '{artist}': {token_id}")
    print(f"Is artist token? {tokenizer.is_artist_token(token_id)}")
    print(f"Artist from token: {tokenizer.get_artist_from_token(token_id)}")

    # Test conditioning
    seq = [100, 200, 300, tokenizer.eos_tok]
    conditioned = tokenizer.add_conditioning(seq, artist)
    print(f"\nOriginal sequence: {seq}")
    print(f"Conditioned sequence: {conditioned}")
    print(f"Removed conditioning: {tokenizer.remove_conditioning(conditioned)}")

    # Test that special tokens are preserved
    print(f"\nEOS token: {tokenizer.eos_tok}")
    print(f"BOS token: {tokenizer.bos_tok}")
    print(f"PAD token: {tokenizer.pad_tok}")


if __name__ == "__main__":
    test_conditional_tokenizer()