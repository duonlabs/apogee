import torch
import huggingface_hub
import numpy as np
import pandas as pd

from typing import Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

from .aggregation import freq2sec, sec2freq
from ..tokenizer import Tokenizer

@dataclass
class DatasetConfig:
    dataset_path: Union[str, Path]
    context_size: int
    start: Optional[int] = None
    end: Optional[int] = None
    temperature: float = 1.4
    max_candles: Optional[int] = None
    deduplicate: bool = False

class CryptoDataset(torch.utils.data.Dataset):
    def __init__(self, metadata: pd.DataFrame, dataset_config: DatasetConfig, tokenizer: Tokenizer):
        self.dataset_config = dataset_config
        self.dataset_path = dataset_config.dataset_path
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.metadata["key"] = metadata.index
        self.metadata["effective_start"] = self.metadata["start"].apply(lambda x: int(max(x, dataset_config.start if dataset_config.start is not None else float("-inf"))))
        self.metadata["effective_end"] = self.metadata["end"].apply(lambda x: int(min(x, dataset_config.end if dataset_config.end is not None else float("inf"))))
        self.metadata["start_offset"] = (self.metadata["effective_start"] - self.metadata["start"]) // self.metadata["freq"]
        self.metadata["end_offset"] = (self.metadata["effective_end"] - self.metadata["start"]) // self.metadata["freq"]
        self.metadata["group"] = self.metadata["key"].apply(lambda x: x.split(".")[1].replace("FDUSD", "USD").replace("USDT", "USD").replace("USDC", "USD"))
        self.metadata = metadata[metadata["effective_end"] > metadata["effective_start"]] # Filter out empty intervals
        if dataset_config.deduplicate:
            self.metadata = self.metadata.loc[self.metadata.groupby("group", sort=False)["effective_start"].idxmin()]
        self.metadata = pd.merge(self.metadata, pd.Series(freq2sec, name="effective_frequency"), how="cross")
        self.metadata["number_of_samples"] = ((self.metadata["effective_end"] - self.metadata["effective_start"]) // self.metadata["effective_frequency"]) // self.dataset_config.context_size
        self.metadata = self.metadata[self.metadata["number_of_samples"] > 0] # Filter out intervals that are too short
        if dataset_config.temperature != 1.0 or dataset_config.max_candles is not None:
            n_samples = np.sum(self.metadata["number_of_samples"].values)
            if dataset_config.max_candles is not None:
                n_samples = min(n_samples, dataset_config.max_candles // dataset_config.context_size)
            logits = np.log(self.metadata["number_of_samples"].values)  # Update to use self.metadata["number_of_samples"]
            target_dist = (np.exp(logits / dataset_config.temperature) / np.sum(np.exp(logits / dataset_config.temperature)))
            factor = target_dist * (n_samples / self.metadata["number_of_samples"].values)
            print("Max repetitions:", np.max(factor))
            repeats = (np.floor(factor).astype(np.int32)) + 1
            self.metadata = self.metadata.iloc[np.repeat(np.arange(len(self.metadata)), repeats)].reset_index(drop=True)
            last_selectors = np.cumsum(repeats) - 1
            self.metadata.loc[np.cumsum(repeats) - 1, "number_of_samples"] = np.round((factor % 1) * self.metadata.loc[last_selectors, "number_of_samples"]).astype(np.int32)
        self.cumulative_samples = np.cumsum(self.metadata["number_of_samples"].values)
        self.length = sum(self.metadata["number_of_samples"].values)
        print("Freq distribution:")
        print(self.metadata[["effective_frequency", "number_of_samples"]].groupby(by=["effective_frequency"], sort=False).sum() / np.sum(self.metadata["number_of_samples"].values))
    @property
    def num_candles(self):
        return self.length * self.dataset_config.context_size

    @property
    def num_tokens(self):
        return self.num_candles * self.tokenizer.tokens_per_candle

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_index = np.searchsorted(self.cumulative_samples, index, side="right")
        pair_start = self.cumulative_samples[pair_index -  1] if pair_index > 0 else 0
        block_index = index - pair_start
        key = self.metadata['key'].values[pair_index]
        secs = self.metadata['effective_frequency'].values[pair_index]
        array = np.load(self.dataset_path / f"{key.replace('.', '/')}.npy", mmap_mode="r")
        array = array[self.metadata["start_offset"].values[pair_index]:self.metadata["end_offset"].values[pair_index]]
        group_size = (secs // self.metadata["freq"].values[pair_index])
        block = array[
            array.shape[0] - (block_index + 1) * self.dataset_config.context_size * group_size:
            array.shape[0] - block_index * self.dataset_config.context_size * group_size
        ]
        block = block.reshape(self.dataset_config.context_size, group_size, 5)
        buffer = np.empty((self.dataset_config.context_size, 5), dtype=np.float32)
        buffer[:, 0] = block[:, 0, 0]
        buffer[:, 1] = np.nanmax(block[..., 1], axis=1)
        buffer[:, 2] = np.nanmin(block[..., 2], axis=1)
        buffer[:, 3] = block[:, -1, 3]
        buffer[:, 4] = np.nansum(block[..., 4], axis=1) if ~np.isnan(block[..., 4]).any() else np.nan
        return self.tokenizer.encode(key, sec2freq[secs], buffer)

    def __len__(self):
        return self.length

@dataclass
class DataloaderConfig:
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = True

@dataclass
class DataConfig:
    hf_repo: str
    cutoff: int
    context_size: int
    revision: Optional[str] = None
    training_temperature: float = 3.0
    val_temperature: float = 1.0
    max_train_candles: Optional[int] = None
    train_deduplicate: bool = False

class DataModule:
    def __init__(self, config: DataConfig, tokenizer: Tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_path = Path(huggingface_hub.snapshot_download(repo_id=config.hf_repo, repo_type="dataset", revision=config.revision))
        self.metadata = pd.read_csv(self.dataset_path / "metadata.csv", index_col="key")
        self.metadata["freq"] = self.metadata["freq"].astype(int)
        self.metadata["start"] = self.metadata["start"].astype(int)
        self.metadata["end"] = self.metadata["end"].astype(int) + self.metadata["freq"]
        self.train_dataset = CryptoDataset(self.metadata, DatasetConfig(self.dataset_path, config.context_size, end=config.cutoff, temperature=config.training_temperature, max_candles=config.max_train_candles, deduplicate=config.train_deduplicate), tokenizer)
        self.val_dataset = CryptoDataset(self.metadata, DatasetConfig(self.dataset_path, config.context_size, start=config.cutoff, temperature=config.val_temperature, deduplicate=False), tokenizer)

    def train_dataloader(self, config: DataloaderConfig):
        return torch.utils.data.DataLoader(self.train_dataset, **config.__dict__)
    
    def val_dataloader(self, config: DataloaderConfig):
        return torch.utils.data.DataLoader(self.val_dataset, **config.__dict__)
