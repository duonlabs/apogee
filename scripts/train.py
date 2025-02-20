import time
import math
import torch
import wandb
import lovely_tensors as lt

from contextlib import nullcontext

from apogee.data.loading import DataModule, DataConfig, DataloaderConfig
from apogee.model import GPT, GPTConfig

lt.monkey_patch()
device = "cuda" if torch.cuda.is_available() else "cpu"

log_2 = torch.tensor(2.0, device=device).log()
@torch.no_grad()
def estimate_metrics(datamodule: DataModule, dataloader_cfg: DataloaderConfig, model: torch.nn.Module, ctx: torch.cuda.amp.autocast, eval_iters: int = 200):
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
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def train_step(
    datamodule: DataModule,
    dataloader_cfg: DataloaderConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.Tensor,
    step: int,
    ctx: torch.cuda.amp.autocast,
    scaler: torch.amp.GradScaler,
    eval_interval: int,
    prof: torch.profiler.profile = None,
):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    t_start = time.time()
    if step % eval_interval == 0:
        metrics = estimate_metrics(datamodule, dataloader_cfg, model, ctx)
        print(f"step {step}: train loss {metrics['train']['loss']:.4f}, val loss {metrics['val']['loss']:.4f}")
        logging = {
            "step": step,
            "lr": lr,  # updated from lr to learning_rate for consistency
        }
        for split in ['train', 'val']:
            for k, v in metrics[split].items():
                logging[f"{split}/{k}"] = v
        wandb.log(logging)
    data = data.to(device)
    X, y = data[:, :-1].long(), data[:, 1:].long()
    with ctx:
        logits = model(X)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    torch.cuda.synchronize()
    if prof: prof.step() #noqa
    step_duration = time.time() - t_start
    print(f"Step: {step}, Loss: {loss.item()}, Duration: {step_duration * 1000:.2f} milliseconds")

if __name__ == '__main__':
    # Args
    hf_repo = 'duonlabs/apogee'
    cutoff = 1730332740
    num_workers = 4
    learning_rate = 1e-3
    batch_size = 32
    profile = False
    eval_iters = 200
    eval_interval = 100
    warmup_iters = 500
    lr_decay_iters = 10000
    min_lr = 1e-5

    # Setup

    wandb.init(project="apogee", name="gpt2", config={
        "cutoff": cutoff,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "profile": profile,
    })

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx =  torch.amp.autocast(device_type=device, dtype=ptdtype)
    datamodule = DataModule(DataConfig(hf_repo=hf_repo, cutoff=cutoff))
    dataloader_cfg = DataloaderConfig(num_workers=num_workers, batch_size=batch_size, shuffle=True)
    model = GPT(GPTConfig(
        n_layer=6,
        # n_layer=12,
    )).to(device)
    model = torch.compile(model)
    model.train()

    dataloader = datamodule.train_dataloader(dataloader_cfg)
    print("Compiling the model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    with (torch.profiler.profile() if profile else nullcontext()) as prof:
        for step, data in zip(range(10 if profile else len(dataloader)), dataloader):
            train_step(datamodule, dataloader_cfg, model, optimizer, data, step, ctx, scaler, eval_interval, prof)
    if prof: 
        prof.export_chrome_trace("trace.json") #noqa
        prof._finalize()