
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "btc_usdt_1h.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

print("=" * 50)
print("CRYPTO DATA VALIDATION")
print("=" * 50)

# 1. Dataset size
print(f"\nTotal candles: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# 2. Missing values
print("\nMissing values:")
print(df.isna().sum())

# 3. Duplicate timestamps
print(f"\nDuplicate timestamps: {df['timestamp'].duplicated().sum()}")

# 4. Chronological order
print(
    "Chronologically sorted:",
    df["timestamp"].is_monotonic_increasing
)

# 5. Missing hourly candles
df = df.sort_values("timestamp").reset_index(drop=True)
time_gaps = df["timestamp"].diff()

gaps = time_gaps[time_gaps > pd.Timedelta(hours=1)]

print(f"Missing-hour gaps: {len(gaps)}")

if not gaps.empty:
    print("\nLargest gaps:")
    print(gaps.sort_values(ascending=False).head())

# 6. Price and volume sanity checks
price_columns = ["open", "high", "low", "close"]

print("\nNon-positive prices:")
print((df[price_columns] <= 0).sum())

print("\nNegative volumes:", (df["volume"] < 0).sum())

invalid_ohlc = (
    (df["high"] < df[["open", "close", "low"]].max(axis=1))
    | (df["low"] > df[["open", "close", "high"]].min(axis=1))
)

print("Invalid OHLC candles:", invalid_ohlc.sum())

print("\nFirst candle:", df["timestamp"].iloc[0])
print("Last candle:", df["timestamp"].iloc[-1])

print("\nValidation complete.")
