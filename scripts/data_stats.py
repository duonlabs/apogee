import argparse
import time
import torch

from tqdm import tqdm

from apogee.data.loading import DataModule, DataConfig, DataloaderConfig

def process_data_stats(hf_repo: str = "duonlabs/apogee", cutoff: int = 1730332740, timeout: int = 60, num_workers: int = 4):
    datamodule = DataModule(DataConfig(hf_repo=hf_repo, cutoff=cutoff))
    print("Dataset stats:")
    print("Train dataset:")
    print(f"Number of tokens: {datamodule.train_dataset.num_tokens / 1_000_000_000:.2f}G")
    print(f"Number of candles: {datamodule.train_dataset.num_candles / 1_000_000:.2f}M")
    print(f"Number of samples: {len(datamodule.train_dataset) / 1_000_000:.2f}M")
    print("Validation dataset:")
    print(f"Number of tokens: {datamodule.val_dataset.num_tokens / 1_000_000_000:.2f}G")
    print(f"Number of candles: {datamodule.val_dataset.num_candles / 1_000_000:.2f}M")
    print(f"Number of samples: {len(datamodule.val_dataset) / 1_000_000:.2f}M")
    print("Total:")
    print(f"Number of tokens: {(datamodule.train_dataset.num_tokens + datamodule.val_dataset.num_tokens) / 1_000_000_000:.2f}G")
    print(f"Number of candles: {(datamodule.train_dataset.num_candles + datamodule.val_dataset.num_candles) / 1_000_000:.2f}M")
    print(f"Number of samples: {(len(datamodule.train_dataset) + len(datamodule.val_dataset)) / 1_000_000:.2f}M")
    dataloader = datamodule.train_dataloader(DataloaderConfig(num_workers=num_workers, shuffle=True))

    start_time = time.time()
    num_tokens = 0
    num_samples = 0
    num_batches = 0
    n_nans_toks = 0
    n_samples_any_nans = 0
    n_samples_all_nans = 0
    progress_bar = tqdm(total=len(dataloader), desc="Processing", unit="batch")
    for batch in dataloader:
        tok_is_nan = (batch[:, 1:].view(batch.shape[0], -1, 20) == torch.tensor([0, 0, 192, 127]*5, dtype=batch.dtype)).all(-1) # [0, 0, 192, 127] is NaN as handled by torch
        n_nans_toks += tok_is_nan.int().sum()
        n_samples_any_nans += tok_is_nan.any(-1).int().sum()
        n_samples_all_nans += tok_is_nan.all(-1).int().sum()
        num_tokens += batch.numel()
        num_samples += batch.shape[0]
        num_batches += 1
        progress_bar.update(1)
        progress_bar.set_postfix(tokens=num_tokens, samples=num_samples, batches=num_batches)
        if time.time() - start_time > timeout:
            break
    progress_bar.close()

    elapsed_time = time.time() - start_time
    toks_per_sec = num_tokens / elapsed_time
    samples_per_sec = num_samples / elapsed_time
    batches_per_sec = num_batches / elapsed_time

    print(f"Processed {num_batches} batches in {elapsed_time:.2f} seconds ({batches_per_sec:.2f} batches/sec)")
    print(f"Processed {num_samples} samples in {elapsed_time:.2f} seconds ({samples_per_sec:.2f} samples/sec)")
    print(f"Processed {num_tokens} tokens in {elapsed_time:.2f} seconds ({toks_per_sec:.2f} tokens/sec)")
    print(f"Number of NaNs toks: {n_nans_toks} ({n_nans_toks / num_samples:.4%})")
    print(f"Number of samples with any NaNs: {n_samples_any_nans} ({n_samples_any_nans / num_samples:.4%})")
    print(f"Number of samples with all NaNs: {n_samples_all_nans} ({n_samples_all_nans / num_samples:.4%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data stats")
    parser.add_argument("--hf_repo", type=str, default="duonlabs/apogee", help="Hugging Face repository")
    parser.add_argument("--cutoff", type=int, default=1730332740, help="Cutoff timestamp")
    parser.add_argument("--timeout", type=int, default=60, help="Time limit for processing data")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader")

    args = parser.parse_args()
    process_data_stats(timeout=args.timeout, hf_repo=args.hf_repo, cutoff=args.cutoff, num_workers=args.num_workers)