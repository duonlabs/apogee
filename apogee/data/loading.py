import torch
import huggingface_hub
import numpy as np
import pandas as pd

from typing import Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

aggregations = {
    "1m": 1 * 60,
    "5m": 5 * 60,
    "30m": 30 * 60,
    "2h": 2 * 60 * 60,
    "8h": 8 * 60 * 60,
    "1d": 24 * 60 * 60,
}

@dataclass
class DatasetConfig:
    dataset_path: Union[str, Path]
    context_size: int = 24
    start: Optional[int] = None
    end: Optional[int] = None
    temperature: float = 1.4

class CryptoDataset(torch.utils.data.Dataset):
    def __init__(self, metadata: pd.DataFrame, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self.dataset_path = dataset_config.dataset_path
        self.metadata = metadata
        self.metadata["key"] = metadata.index
        self.metadata["effective_start"] = self.metadata["start"].apply(lambda x: int(max(x, dataset_config.start if dataset_config.start is not None else float("-inf"))))
        self.metadata["effective_end"] = self.metadata["end"].apply(lambda x: int(min(x, dataset_config.end if dataset_config.end is not None else float("inf"))))
        self.metadata["start_offset"] = (self.metadata["effective_start"] - self.metadata["start"]) // self.metadata["freq"]
        self.metadata["end_offset"] = (self.metadata["effective_end"] - self.metadata["start"]) // self.metadata["freq"]
        self.metadata = metadata[metadata["effective_end"] > metadata["effective_start"]]
        self.metadata = pd.merge(self.metadata, pd.Series(aggregations, name="effective_frequency"), how="cross")
        self.metadata["number_of_samples"] = ((self.metadata["effective_end"] - self.metadata["effective_start"]) // self.metadata["effective_frequency"]) // self.dataset_config.context_size
        if dataset_config.temperature != 1.0:
            logits = np.log(self.metadata["number_of_samples"].values)  # Update to use self.metadata["number_of_samples"]
            factor = (np.exp(logits / dataset_config.temperature) / np.sum(np.exp(logits / dataset_config.temperature))) * (np.sum(self.metadata["number_of_samples"].values) / self.metadata["number_of_samples"].values)
            repeats = (np.floor(factor).astype(np.int32)) + 1
            self.metadata = self.metadata.iloc[np.repeat(np.arange(len(self.metadata)), repeats)].reset_index(drop=True)
            last_selectors = np.cumsum(repeats) - 1
            self.metadata.loc[np.cumsum(repeats) - 1, "number_of_samples"] = np.round((factor % 1) * self.metadata.loc[last_selectors, "number_of_samples"]).astype(np.int32)
        self.cumulative_samples = np.cumsum(self.metadata["number_of_samples"].values)
        self.length = sum(self.metadata["number_of_samples"].values)

    @property
    def num_candles(self):
        return self.length * self.dataset_config.context_size

    @property
    def num_tokens(self):
        return self.length * self.dataset_config.context_size * 4 * 5

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_index = np.searchsorted(self.cumulative_samples, index, side="right")
        pair_start = self.cumulative_samples[pair_index -  1] if pair_index > 0 else 0
        block_index = index - pair_start
        array = np.load(self.dataset_path / f"{self.metadata['key'].values[pair_index].replace('.', '/')}.npy", mmap_mode="r")
        array = array[self.metadata["start_offset"].values[pair_index]:self.metadata["end_offset"].values[pair_index]]
        group_size = (self.metadata["effective_frequency"].values[pair_index] // self.metadata["freq"].values[pair_index])
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
        buffer = buffer.view(np.uint8).reshape(-1).astype(np.uint16)
        buffer = np.concatenate([np.array([256], dtype=np.uint16), buffer]) # Prepend <BOS> (Begin of Series) token
        return torch.tensor(buffer)

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
    revision: Optional[str] = None
    context_size: int = 24
    training_temperature: float = 1.4
    val_temperature: float = 1.0

class DataModule:
    def __init__(self, config: DataConfig):
        self.config = config
        self.dataset_path = Path(huggingface_hub.snapshot_download(repo_id=config.hf_repo, repo_type="dataset", revision=config.revision))
        self.metadata = pd.read_csv(self.dataset_path / "metadata.csv", index_col="key")
        self.metadata["freq"] = self.metadata["freq"].astype(int)
        self.metadata["start"] = self.metadata["start"].astype(int)
        self.metadata["end"] = self.metadata["end"].astype(int) + self.metadata["freq"]
        self.train_dataset = CryptoDataset(self.metadata, DatasetConfig(self.dataset_path, config.context_size, end=config.cutoff, temperature=config.training_temperature))
        self.val_dataset = CryptoDataset(self.metadata, DatasetConfig(self.dataset_path, config.context_size, start=config.cutoff, temperature=config.val_temperature))

    def train_dataloader(self, config: DataloaderConfig):
        return torch.utils.data.DataLoader(self.train_dataset, **config.__dict__)
    
    def val_dataloader(self, config: DataloaderConfig):
        return torch.utils.data.DataLoader(self.val_dataset, **config.__dict__)
