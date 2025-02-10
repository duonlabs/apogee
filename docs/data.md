# Data

## Sourcing

We aim to fetch historical data from major exchanges. We will begin by downloading the **Binance** history for the top pairs as a starting point. It is both convenient and free to access. We will then aim to expand to other exchanges and assets to increase the dataset size. Our approach will include setting up APIs for automation and ensuring data accuracy through validation checks.
We will download candlestick data at a 1-minute interval resolution. We will store the data as numpy buffers with a fixed time stride so that we can easily load it into memory an arbitrary time window.

## Loading

We want to support multi-scale data loading by implementing an efficient dataloader. This dataloader will read contiguous parts of buffers using mmap and perform real time aggregation to recreate frequencies higher than the original resolution. This will allow us to train on different timeframes without having to store multiple copies of the same data.
It will be critical to store the dataset on a SSD to ensure on-the-fly aggregation is fast enough to not become a bottleneck.

## Preprocessing

Once the raw bytes are loaded, we need to handle the preprocessing. This includes fetching of the contextual information and the tokenization step. We will use a simple tokenizer that maps each byte to an integer. We will optionally apply the BPE tokenization to reduce the sequence length and increase the vocabulary size.

## Validation

We will implement a validation pipeline to ensure that there is no look-ahead bias in the data. We will divide the data into training and validation splits with a predefined cutoff timestamp.