import io
import os
import requests
import argparse
import huggingface_hub
import pandas as pd
import numpy as np

from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from apogee.data.sourcing.binance import get_pair_candles

def get_file_from_repo(repo_id: str, filepath: Union[str, Path]) -> bytes:
    response = requests.get(f"https://huggingface.co/datasets/{repo_id}/raw/main/{filepath}")
    response.raise_for_status()
    return response.content

def get_metadata(repo_id: str) -> pd.DataFrame:
    try:
        content = get_file_from_repo(repo_id, "metadata.csv").decode(encoding="utf-8")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            # Create a new metadata file
            content = "key,start,end,freq\n"
            huggingface_hub.create_commit(
                repo_id=repo_id,
                operations=[huggingface_hub.CommitOperationAdd("metadata.csv", content.encode("utf-8"))],
                commit_message="Initialize metadata.csv",
                repo_type="dataset",
            )
        raise e
    return pd.read_csv(io.StringIO(content), index_col="key")

def compute_info_and_buffer(pair: str, n_workers: int = 10) -> Optional[Tuple[str, Dict[str, int], bytes]]:
    try:
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
        npy_buffer = io.BytesIO()
        np.save(npy_buffer, buffer)
        del buffer, df
        return pair, {
            "start": start,
            "end": end,
            "freq": 60,
        }, npy_buffer.getvalue()
    except Exception as e:
        print(f"Failed to compute info and buffer for {pair}: {e}")
        return None

def save_pair_candles(pairs: List[str], repo_id: str = "duonlabs/apogee", n_workers: int = 5, batch_size: int = 5):
    provider = "binance"
    metadata = get_metadata(repo_id)
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        operations = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for computed in executor.map(partial(compute_info_and_buffer, n_workers=n_workers), batch_pairs):
                if computed is None:
                    continue
                pair, info, buffer = computed
                metadata.loc[f"{provider}.{pair}"] = info
                operations.append(huggingface_hub.CommitOperationAdd(f"{provider}/{pair}.npy", buffer))
                print(f"{pair} ready for commit.")
        print(f"Saving {len(operations)} pairs to {repo_id}...")
        huggingface_hub.create_commit(
            repo_id=repo_id,
            operations=operations + [huggingface_hub.CommitOperationAdd("metadata.csv", metadata.to_csv().encode("utf-8"))],
            commit_message=f"Updated {', '.join(batch_pairs)}",
            repo_type="dataset",
        )
        del operations

# Example usage:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and save pair candles data.")
    parser.add_argument("pair", type=str, help="The trading pair to download data for (e.g., BTCUSDT).")
    parser.add_argument("--repo_id", type=str, default="duonlabs/apogee", help="The Hugging Face Hub repository ID to save the data.")
    parser.add_argument("--n_workers", type=int, default=5, help="Number of worker threads to use for downloading data.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of pairs per commit.")
    args = parser.parse_args()
    # Process pair string or file
    if isinstance(args.pair, str):
        if Path(args.pair).exists():
            with open(args.pair, "r") as f:
                args.pair = f.read().strip().splitlines()
        else:
            args.pair = args.pair.split(",")
    # Login to Hugging Face Hub
    huggingface_hub.login(token=os.getenv("HF_TOKEN"))
    # Save pair candles data
    save_pair_candles(args.pair, repo_id=args.repo_id, n_workers=args.n_workers, batch_size=args.batch_size)