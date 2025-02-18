import sys
import pytest
import pandas as pd
import numpy as np

from pathlib import Path

@pytest.fixture
def assets_path() -> Path:
    return Path("tests/assets")

@pytest.fixture
def sample_btc_month_path(assets_path) -> Path:
    return assets_path / "BTCUSDT-1m-2019-03.csv"

@pytest.fixture
def sample_btc_month(sample_btc_month_path) -> pd.DataFrame:
    df = pd.read_csv(sample_btc_month_path, header=None)
    df = df[df.columns[:6]]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = (df["timestamp"] / (np.median(np.diff(df["timestamp"].values)) // 60)).astype(np.uint64)
    return df

def test_buffer_correct(sample_btc_month: pd.DataFrame, btc_buffer: np.ndarray, dataset_metadata: pd.DataFrame):
    btc_metadata = dataset_metadata.loc["binance.BTCUSDT"].to_dict()
    for _, sample_row in sample_btc_month.iterrows():
        sample_row = sample_row.to_dict()
        open, high, low, close, volume = btc_buffer[int((sample_row["timestamp"] - btc_metadata["start"]) // btc_metadata["freq"])]
        assert sample_row["open"] == open
        assert sample_row["high"] == high
        assert sample_row["low"] == low
        assert sample_row["close"] == close
        assert sample_row["volume"] == volume