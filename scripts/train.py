import json
import os
import time
import math
from typing import Callable, Dict, List, Optional, Tuple
import torch
import wandb
import lovely_tensors as lt
import numpy as np

from pathlib import Path
from contextlib import nullcontext
from datetime import datetime
from dataclasses import asdict, dataclass

from apogee.data.loading import DataModule, DataConfig, DataloaderConfig, aggregations
from apogee.model import GPT, ModelConfig

lt.monkey_patch()
device = "cuda" if torch.cuda.is_available() else "cpu"

runs_dir = Path("runs")

@dataclass
class ComputeConfig:
    num_workers: int = 4
    mini_batch_size: int = 32

@dataclass
class Recipe:
    learning_rate: float = 1.2e-3
    min_lr: float = 1e-4
    model_name: str = "gpt2-2.1M"
    data_name: str = "apogee-february-2025"
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    batch_size: int = 32
    warmup_iters: int = 500
    lr_decay_iters: int = 30000

@dataclass
class TrainingSetup:
    recipe_name: str = "gpt2-2.1M-apogee-february-2025"
    profile: bool = False
    eval_iters: int = 200
    eval_interval: int = 1000
    watchlist: Tuple[List[str], List[str]] = (["binance.BTCUSDT", "binance.SOLUSDT", "binance.DOGEUSDT"], ["5m", "8h"])

log_2 = torch.tensor(2.0, device=device).log()
@torch.no_grad()
def estimate_metrics(
    datamodule: DataModule,
    dataloader_cfg: DataloaderConfig,
    model: torch.nn.Module,
    ctx: torch.cuda.amp.autocast,
    eval_iters: int = 200,
    watchlist: Optional[Tuple[List[str], List[str]]] = None,
) -> Dict[str, Dict[str, float]]:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        metrics = {
            "loss": torch.zeros(eval_iters),
            "extropy": torch.zeros(eval_iters),
            "last_candle_extropy": torch.zeros(eval_iters),
        }
        dataloader = getattr(datamodule, f"{split}_dataloader")(dataloader_cfg)
        for k, data in zip(range(eval_iters), dataloader):
            data = data.to(device)
            X, Y = data[:, :-1].long(), data[:, 1:].long()
            with ctx:
                logits = model(X)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none').view(Y.shape[0], -1, 20)
            metrics["loss"][k] = loss.mean().item()
            extropy = (160 - (loss / log_2).sum(-1)) / 160
            metrics["extropy"][k] = extropy.mean().item()
            metrics["last_candle_extropy"][k] = extropy[:, -1].mean().item()
        out[split] = {}
        for k, v in metrics.items():
            out[split][k] = v.mean().item()
    if watchlist is None:
        return out
    out["watchlist"] = {}
    for pair in watchlist[0]:
        assert pair in datamodule.val_dataset.metadata["key"].values
        for freq in watchlist[1]:
            print("Inference for", pair, freq)
            m = datamodule.val_dataset.metadata
            row = m[(m["key"] == pair) & (m["effective_frequency"] == aggregations[freq])].iloc[0]
            idx = row.name.item()
            offset = datamodule.val_dataset.cumulative_samples[idx].item()
            samples = []
            for i in np.random.permutation(datamodule.val_dataset.number_of_samples[idx])[:dataloader_cfg.batch_size]:
                samples.append(datamodule.val_dataset[offset + i])
            data = torch.stack(samples).to(device)
            X, Y = data[:, :-1].long(), data[:, 1:].long()
            with ctx:
                logits = model(X)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none').view(Y.shape[0], -1, 20)
            out["watchlist"][f"{pair}.{freq}.last_candle_extropy"] = ((160 - (loss[:, -1] / log_2).sum(-1)) / 160).mean().item()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr_schedule(recipe: Recipe) -> Callable[[int], float]:
    def get_lr(it) -> float:
        # 1) linear warmup for warmup_iters steps
        if it < recipe.warmup_iters:
            return recipe.learning_rate * (it + 1) / (recipe.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > recipe.lr_decay_iters:
            return recipe.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - recipe.warmup_iters) / (recipe.lr_decay_iters - recipe.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return recipe.min_lr + coeff * (recipe.learning_rate - recipe.min_lr)
    return get_lr

def train_step(
    datamodule: DataModule,
    dataloader_cfg: DataloaderConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.Tensor,
    step: int,
    num_candles: int,
    best_val_loss: float,
    ctx: torch.cuda.amp.autocast,
    scaler: torch.amp.GradScaler,
    training_setup: TrainingSetup,
    recipe: Recipe,
) -> Tuple[int, float]:
    lr = get_lr_schedule(recipe)(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    t_start = time.time()
    if step % training_setup.eval_interval == 0 and step > 0:
        metrics = estimate_metrics(datamodule, dataloader_cfg, model, ctx, watchlist=training_setup.watchlist)
        print(f"step {step}: train loss {metrics['train']['loss']:.4f}, val loss {metrics['val']['loss']:.4f}")
        logging = {
            "step": step,
            "lr": lr,  # updated from lr to learning_rate for consistency
        }
        for split in ['train', 'val'] + (['watchlist'] if training_setup.watchlist is not None else []):
            for k, v in metrics[split].items():
                logging[f"{split}/{k}"] = v
        wandb.log(logging)
        if metrics['val']['loss'] < best_val_loss:
            best_val_loss = metrics['val']['loss']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_config': asdict(model.config),
                'step': step,
                "num_candles": num_candles,
                'best_val_loss': best_val_loss,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    data = data.to(device)
    X, y = data[:, :-1].long(), data[:, 1:].long()
    with ctx:
        logits = model(X)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    num_candles += y.numel() // 20
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if prof: prof.step() #noqa
    step_duration = time.time() - t_start
    if step % 100 == 0:
        print(f"Step: {step}, Loss: {loss.item()}, Duration: {step_duration * 1000:.2f} milliseconds")
    return num_candles, best_val_loss

if __name__ == '__main__':
    # Args
    training_setup = TrainingSetup()

    recipe_path = f"configs/recipes/{training_setup.recipe_name}.json"
    with open(recipe_path, 'r') as f:
        recipe = Recipe(**json.load(f))
    compute_config = ComputeConfig()
    run_name = f"{training_setup.recipe_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(f"configs/models/{recipe.model_name}.json", 'r') as f:
        model_config = ModelConfig(**json.load(f))
    with open(f"configs/data/{recipe.data_name}.json", 'r') as f:
        data_config = DataConfig(**json.load(f))
    
    mup_approx_factor = model_config.mup_base_dim / model_config.n_embd
    learning_rate = recipe.learning_rate * mup_approx_factor
    min_lr = recipe.min_lr * mup_approx_factor
    best_val_loss = float('inf')
    out_dir = runs_dir / run_name
    os.makedirs(out_dir, exist_ok=True)
    # Setup
    wandb.init(project="apogee", name=run_name, config={
        "model_config": asdict(model_config),
        "data_config": asdict(data_config),
        "compute_config": asdict(compute_config),
        "recipe": asdict(recipe),
    })
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx =  torch.amp.autocast(device_type=device, dtype=ptdtype)
    datamodule = DataModule(data_config)
    dataloader_cfg = DataloaderConfig(
        num_workers=compute_config.num_workers,
        batch_size=compute_config.mini_batch_size,
        shuffle=True
    )
    model = GPT(model_config).to(device)
    model = torch.compile(model)
    model.train()

    dataloader = datamodule.train_dataloader(dataloader_cfg)
    print("Compiling the model...")
    optimizer = model.configure_optimizers(learning_rate=learning_rate, weight_decay=recipe.weight_decay, betas=recipe.betas, device_type=device)
    num_candles = 0
    with (torch.profiler.profile() if training_setup.profile else nullcontext()) as prof:
        for step, data in zip(range(10 if training_setup.profile else len(dataloader)), dataloader):
            num_candles, best_val_loss = train_step(datamodule, dataloader_cfg, model, optimizer, data, step, num_candles, best_val_loss, ctx, scaler, training_setup, recipe)
    if prof: 
        prof.export_chrome_trace("trace.json") #noqa
        prof._finalize()