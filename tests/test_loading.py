import pytest
import pandas as pd
import numpy as np

from apogee.tokenizer import Tokenizer
from apogee.data.loading import DataModule, DataConfig

@pytest.fixture
def cutoff() -> int:
    return 1730332740

@pytest.fixture
def context_size() -> int:
    return 24

@pytest.fixture
def tokenizer():
    return Tokenizer()

@pytest.fixture
def datamodule(dataset_name: str, tokenizer: Tokenizer, cutoff: int, context_size: int) -> DataModule:
    return DataModule(DataConfig(hf_repo=dataset_name, cutoff=cutoff, context_size=context_size), tokenizer)

def test_data_module(datamodule: DataModule, tokenizer: Tokenizer, dataset_metadata: pd.DataFrame):
    btcusdt = np.load(datamodule.dataset_path / "binance" / "BTCUSDT.npy").view(np.uint8)
    metadata = dataset_metadata.loc["binance.BTCUSDT"].to_dict()
    cutoff_index=int((datamodule.config.cutoff - metadata["start"]) // metadata["freq"])
    np.testing.assert_equal(btcusdt[cutoff_index-datamodule.config.context_size:cutoff_index].reshape(-1), datamodule.train_dataset[0][tokenizer.meta_context_len:])
    np.testing.assert_equal(btcusdt[-datamodule.config.context_size:].reshape(-1), datamodule.val_dataset[0][tokenizer.meta_context_len:])

def test_too_short_context_bug(datamodule: DataModule):
    assert datamodule.train_dataset[157909].shape[0] == datamodule.config.context_size * 20