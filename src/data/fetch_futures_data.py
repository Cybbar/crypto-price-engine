
from pathlib import Path
import time

import ccxt
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CANDLE_PATH = ROOT / "data" / "futures" / "raw" / "btcusdt_perpetual_1h.csv"
FUNDING_PATH = ROOT / "data" / "futures" / "raw" / "btcusdt_funding.csv"

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "1h"

START_DATE = "2024-01-01T00:00:00Z"

CANDLE_LIMIT = 1000
FUNDING_LIMIT = 1000

SLEEP_SECONDS = 0.2


# ============================================================
# EXCHANGE CONNECTION
# ============================================================

exchange = ccxt.binanceusdm({
    "enableRateLimit": True,
    "timeout": 30000,
})


# ============================================================
# UTILITIES
# ============================================================

def timestamp_to_datetime(timestamp):
    return pd.to_datetime(timestamp, unit="ms", utc=True)


def save_csv_safely(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


# ============================================================
# FETCH FUTURES CANDLES
# ============================================================

def fetch_futures_candles():

    print("\nFetching BTC/USDT perpetual futures candles...")

    since = exchange.parse8601(START_DATE)

    all_rows = []

    while True:

        rows = exchange.fetch_ohlcv(
            SYMBOL,
            timeframe=TIMEFRAME,
            since=since,
            limit=CANDLE_LIMIT,
        )

        if not rows:
            break

        all_rows.extend(rows)

        last_timestamp = rows[-1][0]

        print(
            f"Candles collected: {len(all_rows):,} | "
            f"Latest: {timestamp_to_datetime(last_timestamp)}"
        )

        next_since = last_timestamp + 1

        if next_since <= since:
            raise RuntimeError("Candle pagination did not advance.")

        since = next_since

        if len(rows) < CANDLE_LIMIT:
            break

        time.sleep(SLEEP_SECONDS)

    if not all_rows:
        raise RuntimeError("No futures candles were downloaded.")

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df = pd.DataFrame(all_rows, columns=columns)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df = (
        df.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Exclude the currently forming candle.
    current_hour = pd.Timestamp.now(tz="UTC").floor("h")

    df = df[df["timestamp"] < current_hour].copy()

    if df.empty:
        raise RuntimeError("No completed candles remain after filtering.")

    save_csv_safely(df, CANDLE_PATH)

    print("\nFutures candle download complete.")
    print("Rows:", len(df))
    print("First:", df["timestamp"].min())
    print("Last:", df["timestamp"].max())
    print("Saved:", CANDLE_PATH)

    return df


# ============================================================
# FETCH FUNDING RATES
# ============================================================

def fetch_funding_rates():

    print("\nFetching historical funding rates...")

    since = exchange.parse8601(START_DATE)

    all_rows = []

    while True:

        rows = exchange.fetch_funding_rate_history(
            SYMBOL,
            since=since,
            limit=FUNDING_LIMIT,
        )

        if not rows:
            break

        all_rows.extend(rows)

        last_timestamp = rows[-1]["timestamp"]

        print(
            f"Funding records collected: {len(all_rows):,} | "
            f"Latest: {timestamp_to_datetime(last_timestamp)}"
        )

        next_since = last_timestamp + 1

        if next_since <= since:
            raise RuntimeError("Funding pagination did not advance.")

        since = next_since

        if len(rows) < FUNDING_LIMIT:
            break

        time.sleep(SLEEP_SECONDS)

    if not all_rows:
        raise RuntimeError("No funding records were downloaded.")

    df = pd.DataFrame([
        {
            "timestamp": row["timestamp"],
            "funding_rate": row["fundingRate"],
        }
        for row in all_rows
    ])

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df["funding_rate"] = pd.to_numeric(
        df["funding_rate"],
        errors="coerce",
    )

    df = (
        df.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if df["funding_rate"].isna().any():
        raise RuntimeError("Invalid funding-rate values detected.")

    save_csv_safely(df, FUNDING_PATH)

    print("\nFunding-rate download complete.")
    print("Rows:", len(df))
    print("First:", df["timestamp"].min())
    print("Last:", df["timestamp"].max())
    print("Saved:", FUNDING_PATH)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    exchange.load_markets()

    if SYMBOL not in exchange.markets:
        raise RuntimeError(f"Market unavailable: {SYMBOL}")

    market = exchange.market(SYMBOL)

    if not market.get("swap") or not market.get("linear"):
        raise RuntimeError("Expected a linear perpetual swap market.")

    candles = fetch_futures_candles()
    funding = fetch_funding_rates()

    print("\n====================================")
    print("FUTURES DATA DOWNLOAD SUMMARY")
    print("====================================")
    print("Candles:", len(candles))
    print("Funding records:", len(funding))
    print("Data download finished.")
