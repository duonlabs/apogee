import pytest
import torch
import numpy as np

from apogee.tokenizer import Tokenizer

@pytest.fixture
def tokenizer():
    return Tokenizer()

def test_tokenizer_reversible(tokenizer: Tokenizer, btc_buffer: np.array):
    tokens = tokenizer.encode(btc_buffer)
    candles = tokenizer.decode(tokens)
    assert (torch.tensor(btc_buffer).view(torch.uint32) == candles.view(torch.uint32)).all()