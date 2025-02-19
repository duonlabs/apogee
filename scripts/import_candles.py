import argparse
import pandas as pd
import numpy as np

from pathlib import Path
from apogee.data.sourcing.binance import get_pair_candles

def save_pair_candles(pair: str, n_workers: int = 10, dataset_path: Path = Path("dataset")):
    provider_path = dataset_path / "binance"
    provider_path.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_path / "metadata.csv"
    metadata = pd.read_csv(metadata_path, index_col="key")
    df = get_pair_candles(pair, n_workers)
    timestamps = df["timestamp"].values
    start = timestamps[0]
    end = timestamps[-1]
    df = df.drop(columns=["timestamp"])
    buffer = np.full(((end-start) // 60 + 1, 5), np.nan, dtype=np.float32)
    i_start = 0
    for i_end in (np.where(np.diff(timestamps)!=60)[0]+1).tolist() + [len(timestamps)]:
        ts_start, ts_end = timestamps[i_start], timestamps[i_end-1]
        buffer[(ts_start - start) // 60:1+(ts_end - start) // 60] = df.values[i_start:i_end]
        if i_end < len(timestamps):
            i_start = i_end + 1 if (timestamps[i_end]-timestamps[i_end-1])<60 else i_end
    key = f"binance.{pair}"
    metadata.loc[key] = {
        "start": start,
        "end": end,
        "freq": 60,
    }
    metadata.to_csv(metadata_path)
    np.save(provider_path / f"{pair}.npy", buffer.view(np.uint8))

# Example usage:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and save pair candles data.")
    parser.add_argument("pair", type=str, help="The trading pair to download data for (e.g., BTCUSDT).")
    parser.add_argument("--n_workers", type=int, default=10, help="Number of worker threads to use for downloading data.")
    args = parser.parse_args()
    
    save_pair_candles(args.pair, args.n_workers)