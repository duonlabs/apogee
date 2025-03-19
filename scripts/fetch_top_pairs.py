import time
import argparse
import requests
import csv

def fetch_pairs(exchange: str) -> list:
    pairs = []
    page = 1
    done = False
    while not done:
        print(f'Fetching page {page} of {exchange} pairs...', end='\r')
        response = requests.get(
            f'https://api.coingecko.com/api/v3/exchanges/{exchange}/tickers',
            params={'page': page}
        )
        response.raise_for_status()
        data = response.json()
        if 'tickers' not in data or not data['tickers']:
            done = True
            break
        pairs.extend(data['tickers'])
        page += 1
        time.sleep(20)
    return pairs

def get_nested_value(d, keys):
    for key in keys:
        d = d.get(key, {})
    return d

def main():
    parser = argparse.ArgumentParser(description='Fetch and save top trading pairs from an exchange.')
    parser.add_argument('--exchange', type=str, default="binance", help='The exchange to fetch pairs from')
    parser.add_argument('--num_pairs', type=int, default=None, help='Number of pairs to fetch')
    parser.add_argument('--sort_key', type=str, default=None, help='Key to sort pairs by')
    args = parser.parse_args()

    pairs = fetch_pairs(args.exchange)
    if args.sort_key is not None:
        pairs = sorted(pairs, key=lambda x: get_nested_value(x, args.sort_key.split('.')), reverse=True)
    if args.num_pairs is not None:
        pairs = pairs[:args.num_pairs]
    
    with open(f'{args.exchange}.txt', 'w') as f:
        f.write("\n".join(map(lambda x: f"{x["base"]}{x["target"]}", pairs)))

if __name__ == '__main__':
    main()
