import time
import torch

from contextlib import nullcontext
from typing import Any, Dict, Optional, Union
from pathlib import Path

from apogee.tokenizer import Tokenizer
from apogee.model import GPT, ModelConfig

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

class EndpointHandler:
    """
    Handler class.
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        if base_path is None:
            base_path = Path(__file__).parent
        self.base_path = Path(base_path)
        # Get the device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Handler spwaned on device {self.device} 🚀")
        ckpt_path = self.base_path / "ckpt.pt"
        print(f"Loading model from {ckpt_path} 🤖")
        checkpoint = torch.load(ckpt_path, map_location=device)
        self.config = ModelConfig(**checkpoint["model_config"])
        self.tokenizer = Tokenizer()
        self.model = GPT(self.config)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        self.model = torch.compile(self.model)
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8 else 'float16' # 'float32' or 'bfloat16' or 'float16'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
        print("Warming up hardware 🔥")
        with torch.no_grad(), self.ctx:
            self.model(torch.randint(0, self.tokenizer.vocabulary_size, (1, self.config.block_size), device=self.device))
        print("Model ready ! ✅")
        # Precompute useful values
        self.max_candles = (self.config.block_size - self.config.meta_size) // self.tokenizer.tokens_per_candle

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            data (Dict[str, Any]):
                inputs: Dict[str, Union[str, List[float]]] with keys:
                    pair: Pair symbol
                    frequency: Frequency of the time serie (1m, 5m, 30m, 2h, 8h, 1d)
                    timestamps: Timestamps of the time serie
                    open: Open prices
                    high: High prices
                    low: Low prices
                    close: Close prices
                    volume: Volumes
                steps: int = 4 | Number of sampling steps
                n_scenarios: int = 32 | Number of scenarios to generate
                seed: Optional[int] = None | Seed for the random number generator
        Return:
            Dict[str, Any] Generated scenarios with keys:
                timestamps: Timestamps of the time serie
                open: Open prices
                high: High prices
                low: Low prices
                close: Close prices
                volume: Volumes
        """
        t_start = time.time() # Start the timer
        # Unpack input data
        inputs = data.pop("inputs", data)
        # Validate the inputs
        assert "pair" in inputs and "frequency" in inputs and "timestamps" in inputs and "open" in inputs and "high" in inputs and "low" in inputs and "close" in inputs and "volume" in inputs, "Required keys: pair, frequency, timestamps, open, high, low, close, volume"
        assert isinstance(inputs["pair"], str) and isinstance(inputs["frequency"], str) and isinstance(inputs["timestamps"], list) and isinstance(inputs["open"], list) and isinstance(inputs["high"], list) and isinstance(inputs["low"], list) and isinstance(inputs["close"], list) and isinstance(inputs["volume"], list), "Inputs must be lists"
        assert inputs["frequency"] in ["1m", "5m", "30m", "2h", "8h", "1d"], "Invalid frequency"
        assert len(inputs["timestamps"]) == len(inputs["open"]) == len(inputs["high"]) == len(inputs["low"]) == len(inputs["close"]) == len(inputs["volume"]), "Inputs must have the same length"
        pair, freq = inputs["pair"], inputs["frequency"]
        pair = "".join(pair.split("/"))
        pair = f"binance.{pair.upper()}" if "." not in pair else pair
        timestamps = torch.tensor(inputs["timestamps"])
        samples = torch.tensor([inputs["open"], inputs["high"], inputs["low"], inputs["close"], inputs["volume"]], dtype=torch.float32).T.contiguous()
        steps = data.pop("steps", 4)
        n_scenarios = data.pop("n_scenarios", 32)
        seed = data.pop("seed", None)
        # Validate the params
        assert isinstance(steps, int) and steps > 0, "steps must be a positive integer"
        assert isinstance(n_scenarios, int) and n_scenarios > 0, "n_scenarios must be a positive integer"
        if seed is not None:
            assert isinstance(seed, int), "seed must be an integer"
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        # Generate scenarios
        samples = samples[-self.max_candles + steps:] # Keep only the last candles that fit in the model's context
        tokens = self.tokenizer.encode(pair, freq, samples) # Encode the samples into tokens
        tokens = tokens.to(self.device).unsqueeze(0).long() # Add a batch dimension
        with torch.no_grad(), self.ctx:
            for _ in range(steps * self.tokenizer.tokens_per_candle):
                assert tokens.shape[1] <= self.config.block_size, "Too many tokens in the sequence"
                logits = self.model(tokens) # forward the model to get the logits for the index in the sequence
                logits = logits[:, -1, :] # pluck the logits at the final step
                # apply softmax to convert logits to (normalized) probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                # sample from the distribution
                if probs.shape[0] != n_scenarios:
                    next_tokens = torch.multinomial(probs, num_samples=n_scenarios, replacement=True).T
                    tokens = tokens.expand(n_scenarios, -1)
                else:
                    next_tokens = torch.multinomial(probs, num_samples=1)
                # append sampled index to the running sequence and continue
                tokens = torch.cat((tokens, next_tokens), dim=1)
        # Decode the tokens back into samples
        _, _, scenarios = self.tokenizer.decode(tokens)
        scenarios = scenarios[:, -steps:]
        print(f"Generated {n_scenarios} scenarios in {time.time() - t_start:.2f} seconds ⏱")
        return {
            "timestamps": (timestamps[-1] + torch.arange(1, steps+1) * torch.median(torch.diff(timestamps)).item()).tolist(),
            "open": scenarios[:, :, 0].tolist(),
            "high": scenarios[:, :, 1].tolist(),
            "low": scenarios[:, :, 2].tolist(),
            "close": scenarios[:, :, 3].tolist(),
            "volume": scenarios[:, :, 4].tolist()
        }

if __name__ == "__main__":
    import pandas as pd
    handler = EndpointHandler()
    test_path = Path(__file__).parents[2] / "tests" / "assets" / "BTCUSDT-1m-2019-03.csv"
    with open(test_path, "r") as f:
        data = pd.read_csv(f)
    y = handler({
        "inputs": {
            "pair": "binance.BTCUSDT",
            "frequency": "1m",
            "timestamps": data[data.columns[0]].tolist(),
            "open": data[data.columns[1]].tolist(),
            "high": data[data.columns[2]].tolist(),
            "low": data[data.columns[3]].tolist(),
            "close": data[data.columns[4]].tolist(),
            "volume": data[data.columns[5]].tolist()
        },
        "steps": 4,
        "n_scenarios": 64,
        "seed": 42
    })