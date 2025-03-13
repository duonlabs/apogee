import sys
import torch

import numpy as np

from typing import List, Tuple, Union
from .data.aggregation import freq2sec

class Tokenizer:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 " # Allowed characters in pair names
    freqs = [k for k, _ in sorted(freq2sec.items(), key=lambda x: x[1])] # Sort the freq2sec by duration and extract the keys
    pair_name_max_len: int = 10
    vocabulary_size: int = 256 + 1 + len(letters) + len(freqs) # 256 possible bytes + 1 for <BOS> token + len(letters) for pair name
    tokens_per_candle: int = 4*5
    meta_context_len: int = 11

    def encode(self, key: str, freq: str, candles: Union[np.array, torch.Tensor]) -> torch.Tensor:
        """Tokenize candles into tokens."""
        _, pair = key.split(".") # Split the key into exchange and pair
        meta = torch.tensor([257 + Tokenizer.letters.index(letter) for letter in pair.upper().ljust(self.pair_name_max_len)] + [257 + len(Tokenizer.letters) + Tokenizer.freqs.index(freq)], dtype=torch.uint16) # Encode the pair name and frequency
        if isinstance(candles, np.ndarray): # Wrap into a tensor
            candles = torch.tensor(candles)
        candles = (candles.view(torch.int32) << 1).view(torch.float32) # Erase the sign bit to fit the exponent into the first byte
        if sys.byteorder == 'little': # On little-endian systems, we need to byteswap the data so that msb is first
            candles.untyped_storage().byteswap(torch.float32)
        buffer = candles.view(torch.uint8) # Interpret the data as bytes ("tokenization" step)
        buffer = buffer.view(-1).to(torch.uint16) # Flatten the data and convert to uint16 because otherwise <BOS> will overflow
        buffer = torch.cat([meta, torch.tensor([256], dtype=torch.uint16), buffer]) # Prepend <BOS> (Begin of Series) token
        return buffer
    
    def decode(self, tokens: torch.Tensor) -> Tuple[Union[List[str], str], Union[List[str], str], torch.Tensor]:
        """Decode tokens into candles."""
        tokens = tokens.long()
        meta_tokens, candles_tokens = tokens[..., :Tokenizer.meta_context_len], tokens[..., Tokenizer.meta_context_len + 1:] # Remove <BOS> token
        candles_tokens = candles_tokens.to(torch.uint8).view(*tokens.shape[:-1], -1, self.tokens_per_candle) # Convert back to uint8 and reshape
        candles_tokens = candles_tokens.view(torch.float32) # Interpret the data as floats
        if sys.byteorder == 'little': # On little-endian systems, we need to byteswap the data back
            # candles_tokens.untyped_storage().byteswap(torch.float32) # <-- This segfaults for some reason
            candles_tokens = candles_tokens.view(torch.uint8).view(*candles_tokens.shape, 4).flip(-1).view(torch.float32).squeeze(-1)# Workaround
        candles_tokens = -((candles_tokens.view(torch.int32) >> 1) | (1 << 31)).view(torch.float32) # Restore the sign bit
        if meta_tokens.ndim == 1:
            meta_tokens = meta_tokens.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        pair_meta, freq_meta = meta_tokens[..., :-1], meta_tokens[..., -1] # Extract pair and frequency tokens
        pairs = []
        freqs = []
        for i in range(len(pair_meta)):
            pairs.append("".join(Tokenizer.letters[token-257] for token in pair_meta[i].tolist()).rstrip(" "))
            freqs.append(Tokenizer.freqs[freq_meta[i] - (257 + len(Tokenizer.letters))])
        pair = pairs[0] if squeeze else pairs
        freq = freqs[0] if squeeze else freqs
        return pair, freq, candles_tokens