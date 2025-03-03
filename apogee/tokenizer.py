import sys
import torch

import numpy as np

from typing import Union

class Tokenizer:
    @property
    def vocabulary_size(self) -> int:
        """Return the size of the vocabulary"""
        return 257
    
    @property
    def tokens_per_candle(self) -> int:
        """Return the number of tokens per candle"""
        return 4 * 5

    def encode(self, candles: Union[np.array, torch.Tensor]) -> torch.Tensor:
        """Tokenize candles into tokens."""
        if isinstance(candles, np.ndarray):
            candles = torch.tensor(candles)
        if sys.byteorder == 'little':
            candles.untyped_storage().byteswap(torch.float32)
        buffer = candles.view(torch.uint8)
        buffer = buffer.view(-1).to(torch.uint16)
        buffer = torch.cat([torch.tensor([256], dtype=torch.uint16), buffer]) # Prepend <BOS> (Begin of Series) token
        return buffer
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens into candles."""
        tokens = tokens.long()
        candles_tokens = tokens[..., 1:]
        candles_tokens = candles_tokens.to(torch.uint8).view(*tokens.shape[:-1], -1, self.tokens_per_candle)
        candles_tokens = candles_tokens.view(torch.float32)
        if sys.byteorder == 'little':
            # candles_tokens.untyped_storage().byteswap(torch.float32) # <-- This segfaults for some reason
            candles_tokens = candles_tokens.view(torch.uint8).view(*candles_tokens.shape, 4).flip(-1).view(torch.float32).squeeze(-1)
        return candles_tokens