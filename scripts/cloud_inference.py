import ccxt
import os
import requests
import numpy as np

API_URL = os.getenv('HF_API_URL')
headers = {
	"Accept" : "application/json",
	"Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
	"Content-Type": "application/json" 
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

exchange_name = 'binance'
exchange = ccxt.binance()
def get_last_crypto_data(symbol, timeframe, limit):
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit + 1)
    data = np.array(data)[:-1]
    return data

pair = 'BTC/USDT'
freq = '8h'
limit = 48
print(f"Fetching last {limit} {freq} candles for {pair}")
data = get_last_crypto_data(pair, freq, limit)
print("Querying the model")
response = query({
    "inputs": {
        "timestamps": data[:, 0].tolist(),
        "open": data[:, 1].tolist(),
        "high": data[:, 2].tolist(),
        "low": data[:, 3].tolist(),
        "close": data[:, 4].tolist(),
        "volume": data[:, 5].tolist(),
    },
    "steps": 4,
    "n_scenarios": 128,
    "seed": 42
})

response = np.array([response["open"], response["high"], response["low"], response["close"], response["volume"]]).transpose(1, 2, 0)
print(response)
breakpoint()