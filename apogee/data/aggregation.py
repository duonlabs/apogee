

freq2sec = {
    "1m": 1 * 60,
    "5m": 5 * 60,
    "30m": 30 * 60,
    "2h": 2 * 60 * 60,
    "8h": 8 * 60 * 60,
    "1d": 24 * 60 * 60,
}

sec2freq = {v: k for k, v in freq2sec.items()}