import sys
import pytest
import huggingface_hub
import pandas as pd
import numpy as np

from pathlib import Path

sys.path.append(".")

@pytest.fixture
def dataset_name() -> str:
    return "duonlabs/apogee"

@pytest.fixture
def dataset_path(dataset_name: str) -> Path:
    return Path(huggingface_hub.snapshot_download(repo_id=dataset_name, repo_type="dataset"))

@pytest.fixture
def btc_buffer_path(dataset_path) -> Path:
    return dataset_path / "binance" / "BTCUSDT.npy"

@pytest.fixture
def btc_buffer(btc_buffer_path) -> np.ndarray:
    return np.load(btc_buffer_path).view(np.float32)

@pytest.fixture
def dataset_metadata_path(dataset_path) -> Path:
    return dataset_path / "metadata.csv"

@pytest.fixture
def dataset_metadata(dataset_metadata_path) -> pd.DataFrame:
    return pd.read_csv(dataset_metadata_path, index_col="key") 
    