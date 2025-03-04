from contextlib import nullcontext
import torch

from apogee.data.loading import DataModule, DataConfig, aggregations
from apogee.tokenizer import Tokenizer
from apogee.model import GPT, ModelConfig

torch.set_printoptions(precision=4, sci_mode=False)

hf_repo = 'duonlabs/apogee'
cutoff = 1730332740
coin="binance.BTCUSDT"
agg = "8h"
tokenizer = Tokenizer()
datamodule = DataModule(DataConfig(hf_repo=hf_repo, cutoff=cutoff), tokenizer)
df=datamodule.val_dataset.metadata
row = df[(df["key"]==coin)&(df["effective_frequency"]==aggregations[agg])]
offset=datamodule.val_dataset.cumulative_samples[row.iloc[0].name.item()].item()
num_samples = 10 # number of samples to draw
candles_horizon = 1 # number of tokens generated in each sample
tokens_horizon = candles_horizon * tokenizer.tokens_per_candle
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = None # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 42
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8 else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "runs/gpt2-2.4M-apogee-february-2025-20250303_101748/ckpt.pt"

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# model
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = ModelConfig(**checkpoint["model_config"])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

data = datamodule.val_dataset[offset+0]
start_ids = data[:-tokens_horizon]
print("True continuation:")
print(tokenizer.decode(data[-tokens_horizon:]))
print('---------------')
x = start_ids.long().to(device)[None, ...]
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, tokens_horizon, temperature=temperature, top_k=top_k)
            candle = tokenizer.decode(y[0, -tokens_horizon:])
            print(candle)
            if torch.max(candle[:4]).item() != candle[1].item():
                print("Warning: Max value does not equal the second candle value.")
            if torch.min(candle[:4]).item() != candle[2].item():
                print("Warning: Min value does not equal the third candle value.")
            print('---------------')
