# Apogée: Scaling Laws for Crypto Market Forecasting

## Introduction

Apogée is an open-source research initiative exploring the scaling laws of crypto market forecasting. While financial markets are often deemed unpredictable, deep learning has repeatedly demonstrated that performance scales with compute and data. Our goal is to determine whether this principle holds for large models trained on historical crypto candlestick data.

## The Core Question

Instead of asking, *"Can the market be predicted?"*, we ask:
*"How predictable is the market?"*

We measure predictability in **bits** of future price information inferable from past data, systematically increasing model size, dataset scope, and compute budget to uncover whether a scaling law emerges.

## Research Objectives

### 1. Quantifying Predictability

We train deep learning models to predict next-candles autoregressively, measuring how many bits of the five float32 values (Open, High, Low, Close, Volume) can be inferred purely from past candlestick history.

### 2. Establishing Market Scaling Laws

Similar to how LLMs improve predictability with scaling, we examine whether financial time-series models show analogous patterns.

**Hypothesis**: Increasing dataset size (longer histories, more assets) and model capacity (larger neural nets) improves predictive power.

Outcome Possibilities:

* 📉 A strict upper bound emerges (then we want to quantify it)

* 📈 Performance scales steadily (EMH reformulatioon as a function of flops).

### 3. Assessing the Efficient Market Hypothesis (EMH)

The EMH suggests markets incorporate all information instantly, making future price prediction impossible. If true, then scaling deep learning models should yield diminishing returns. But if predictability continues improving with more compute, we may expose an explanatory gap in the EMH.

## How It Works

### 🗄 Dataset & Preprocessing

Fetching historical candlestick data from major exchanges.

Implementing high-performance data pipelines optimized for large-scale training.

Preventing look-ahead bias with rigorous time-series validation.

More details in the [data.md](docs/data.md) file.

### 🤖 Modeling Approaches

Using a variant of the Byte Latent Transformer (BLT) architecture.

Experimenting with Transformers, State-Space Models (SSMs), and other attention-free backbones.

Using minimal contextual information (pair name, timeframe) to guide predictions.

Optimizing for efficiency with domain-specific GPU kernels.

More details in the [model.md](docs/model.md) file.

### 🚀 Scaling Strategy

Gradually increasing compute budget and dataset size.

Measuring per FLOP efficiency of training larger models vs. longer training cycles.

Investigating whether performance follows a power-law curve or plateaus.

## Open Source from Day One

Apogée adheres to a strict open-source ethos:
* Code & Pipelines - All preprocessing scripts, model definitions, and training routines will be fully public.
* Reproducible Results - Logs, hyperparameters, and hardware configurations will be documented for replication.
* Community Benchmark - Our results will define a baseline for market predictability, enabling others to extend or challenge our findings.

## Roadmap

### Software Development Phase
- 🚧 Data sourcing
    - Code ✅
    - Tests 🚧
    - Re-import all of Binance without tokenization 🔜
    - Import all of Binance 🔜
    - Add providers 🔜
- 🚧 Data loading
    - Code ✅
    - Tests 🔜
- 🔜 Models
- 🔜 Training
- 🔜 Evaluation
- 🔜 Benchmarking

### Research Phase
- 🔜 Baseline performances
- 🔜 Scaling experiments
- 🔜 Scaling laws analysis
- 🔜 Paper drafting

### Speedrun Phase
- Create a minimal script inspired by the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt)
- Bounties for speedrunning milestones

## Get Involved

We’re looking for technically-minded collaborators who share our vision. If you're interested in contributing, we welcome:
- Contributors - Deep learning experts, quants, and engineers.
- Supporters - Cloud providers and GPU sponsors for large-scale training.
- Funding - Research grants or investments to help scale our work and pay contributors.

📌 [Join the discussion](https://t.me/DuonLabs)

By measuring the measurable, Apogée will either confirm market randomness or uncover a new frontier in financial time-series modeling. Let's push the limits of predictability together.

