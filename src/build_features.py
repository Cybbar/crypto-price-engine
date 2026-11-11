
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "data" / "raw" / "btc_usdt_1h.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "btc_features.csv"


def build_features():
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Price movement
    df["return_1h"] = np.log(df["close"] / df["close"].shift(1))
    df["return_24h"] = np.log(df["close"] / df["close"].shift(24))

    # Volatility based only on past and current hourly returns
    df["volatility_24h"] = (
        df["return_1h"].rolling(24).std()
    )

    # Trading volume activity relative to the previous 24 hours.
    # Shift the rolling baseline so current volume isn't in its own baseline.
    previous_volume_avg = (
        df["volume"].rolling(24).mean().shift(1)
    )
    df["volume_ratio_24h"] = df["volume"] / previous_volume_avg

    # Current candle's range relative to its closing price
    df["range_ratio"] = (df["high"] - df["low"]) / df["close"]

    # Price compared with the previous 24 hours' average close
    previous_price_avg = (
        df["close"].rolling(24).mean().shift(1)
    )
    df["trend_24h"] = df["close"] / previous_price_avg - 1

    # Prediction targets: future returns.
    # These are labels, NOT model input features.
    df["target_return_1h"] = np.log(
        df["close"].shift(-1) / df["close"]
    )
    df["target_return_24h"] = np.log(
        df["close"].shift(-24) / df["close"]
    )

    # Remove rows without enough historical data or future targets.
    feature_columns = [
        "return_1h",
        "return_24h",
        "volatility_24h",
        "volume_ratio_24h",
        "range_ratio",
        "trend_24h",
    ]

    target_columns = [
        "target_return_1h",
        "target_return_24h",
    ]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=feature_columns + target_columns).copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Raw candles: {len(pd.read_csv(INPUT_FILE)):,}")
    print(f"Feature rows: {len(df):,}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nFeature preview:")
    print(
        df[
            ["timestamp"] + feature_columns + target_columns
        ].tail().to_string(index=False)
    )

    print("\nMissing values in feature columns:")
    print(df[feature_columns + target_columns].isna().sum())


if __name__ == "__main__":
    build_features()
