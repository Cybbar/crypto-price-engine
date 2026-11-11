
import time
from pathlib import Path

import ccxt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_DIR / "data" / "raw" / "btc_usdt_1h.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

exchange = ccxt.binance({
    "enableRateLimit": True,
    "timeout": 30000,
})

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
START_DATE = "2024-01-01T00:00:00Z"
LIMIT = 1000


def fetch_data():
    since = exchange.parse8601(START_DATE)
    candles_all = []

    print(f"Fetching {SYMBOL} hourly candles from Binance...")

    while True:
        try:
            candles = exchange.fetch_ohlcv(
                SYMBOL,
                timeframe=TIMEFRAME,
                since=since,
                limit=LIMIT,
            )
        except Exception as error:
            print(f"\nDownload failed: {error}")
            print("Check your internet connection and try again.")
            return

        if not candles:
            break

        candles_all.extend(candles)

        next_since = candles[-1][0] + 3_600_000
        if next_since <= since:
            break

        since = next_since

        print(f"Collected {len(candles_all)} candles so far.")

        if len(candles) < LIMIT:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not candles_all:
        print("No candles received.")
        return

    columns = [
        "timestamp", "open", "high", "low", "close", "volume"
    ]
    df = pd.DataFrame(candles_all, columns=columns)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    )

    df = (
        df.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Exclude the current, unfinished hourly candle.
    current_hour = pd.Timestamp.now(tz="UTC").floor("h")
    df = df[df["timestamp"] < current_hour]

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(df)} candles.")
    print(f"File: {OUTPUT_FILE}")
    print("\nLatest five candles:")
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    fetch_data()
