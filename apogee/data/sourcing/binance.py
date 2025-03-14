import io
import requests
import xml.etree.ElementTree as ET
import zipfile
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs
from datetime import datetime

def get_prefix_from_website_url(url: str, s3b_root_dir: str = 'data/') -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    prefix = qs.get('prefix', [s3b_root_dir])[0]
    if not prefix.startswith(s3b_root_dir):
        prefix = s3b_root_dir + prefix
    if not prefix.endswith('/'):
        prefix += '/'
    return prefix

def create_s3_query_url(bucket_url: str, prefix: str, marker: str = None) -> str:
    url = bucket_url + '?delimiter=/'
    url += '&prefix=' + prefix
    if marker:
        url += '&marker=' + marker
    return url

def bytes_to_human_readable(size_in_bytes: str) -> str:
    size = float(size_in_bytes)
    i = -1
    units = [' kB', ' MB', ' GB']
    while size > 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    if i == -1:
        return f"{size_in_bytes} bytes"
    return f"{max(size, 0.1):.1f}{units[i]}"

def get_info_from_s3_data(xml_text: str) -> dict:
    # Define the namespace from the XML
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(xml_text)
    
    prefix_elem = root.find('s3:Prefix', ns)
    prefix = prefix_elem.text if prefix_elem is not None else ''
    
    files = []
    for content in root.findall('s3:Contents', ns):
        key_elem = content.find('s3:Key', ns)
        if key_elem is None:
            continue
        key = key_elem.text
        last_modified = content.find('s3:LastModified', ns).text if content.find('s3:LastModified', ns) is not None else ''
        size_val = content.find('s3:Size', ns).text if content.find('s3:Size', ns) is not None else '0'
        files.append({
            'Key': key,
            'LastModified': last_modified,
            'Size': bytes_to_human_readable(size_val),
            'Type': 'file'
        })
    # Remove first entry if it equals the prefix (as in JS)
    if prefix and files and files[0]['Key'] == prefix:
        files.pop(0)
    
    directories = []
    for cp in root.findall('s3:CommonPrefixes', ns):
        pfx_elem = cp.find('s3:Prefix', ns)
        pfx = pfx_elem.text if pfx_elem is not None else ''
        directories.append({
            'Key': pfx,
            'LastModified': '',
            'Size': '',
            'Type': 'directory'
        })
    
    is_truncated_elem = root.find('s3:IsTruncated', ns)
    is_truncated = (is_truncated_elem.text.lower() == 'true') if is_truncated_elem is not None else False
    next_marker = None
    if is_truncated:
        next_marker_elem = root.find('s3:NextMarker', ns)
        next_marker = next_marker_elem.text if next_marker_elem is not None else None
        
    return {
        'files': files,
        'directories': directories,
        'prefix': prefix,
        'nextMarker': next_marker
    }

def get_s3_data(bucket_url: str, prefix: str, marker: str = None) -> dict:
    url = create_s3_query_url(bucket_url, prefix, marker)
    response = requests.get(url)
    response.raise_for_status()
    info = get_info_from_s3_data(response.text)
    if info['nextMarker']:
        next_info = get_s3_data(bucket_url, prefix, info['nextMarker'])
        info['files'].extend(next_info['files'])
        info['directories'].extend(next_info['directories'])
    return info

def load_zip_file(url: str) -> pd.DataFrame:
    content = requests.get(url).content
    print(f"Downloaded {len(content)} bytes from {url}")
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for f in z.filelist:
            with z.open(f.filename) as zf:
                df = pd.read_csv(zf, header=None)
                df = df[df.columns[:6]]
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
                df["timestamp"] = (df["timestamp"] / (np.median(np.diff(df["timestamp"].values)) // 60)).astype(np.uint64)
                return df

def get_pair_candles(pair: str, n_workers: int = 10) -> pd.DataFrame:
    base_url = "https://data.binance.vision/"
    website_url = f'{base_url}?prefix=data/spot/monthly/klines/{pair}/1m/'
    s3b_root_dir = 'data/'
    prefix = get_prefix_from_website_url(website_url, s3b_root_dir)
    
    # Binance S3 API endpoint
    bucket_url = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
    
    listing = get_s3_data(bucket_url, prefix)
    files = list(filter(lambda x: not x["Key"].endswith("CHECKSUM"), listing["files"]))
    keys = map(lambda x: x["Key"].split("/")[-1].split(".")[0], files)
    keys = map(lambda x: datetime(int(x.split("-")[-2]), int(x.split("-")[-1]), 1), keys)
    values = map(lambda x: base_url + x["Key"], files)
    files = dict(zip(keys, values))
    dfs = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        dfs = list(executor.map(load_zip_file, sorted(files.values())))
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    df.index = df["timestamp"]
    return df.sort_index()