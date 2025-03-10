import pytest
import torch
import numpy as np

from apogee.tokenizer import Tokenizer

@pytest.fixture
def tokenizer():
    return Tokenizer()

def test_tokenizer_reversible(tokenizer: Tokenizer, btc_buffer: np.array):
    tokens = tokenizer.encode("binance.BTCUSDT", "1m", btc_buffer)
    pair, freq, candles = tokenizer.decode(tokens)
    assert pair == "BTCUSDT"
    assert freq == "1m"
    assert (torch.tensor(btc_buffer).view(torch.uint32) == candles.view(torch.uint32)).all()