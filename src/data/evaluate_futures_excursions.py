from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CANDLE_FILE = (
    ROOT / "data" / "futures" / "raw"
    / "btcusdt_perpetual_1h.csv"
)

REGIME_FILE = (
    ROOT / "data" / "futures" / "processed"
    / "btcusdt_futures_regimes.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "futures" / "processed"
    / "futures_excursion_evaluation.csv"
)

HORIZON = 24


def evaluate_excursions():
    candles = pd.read_csv(
        CANDLE_FILE,
        parse_dates=["timestamp"],
    )

    regimes = pd.read_csv(
        REGIME_FILE,
        parse_dates=["timestamp"],
    )

    candles = candles.sort_values("timestamp").copy()
    regimes = regimes.sort_values("timestamp").copy()

    required_candles = ["timestamp", "high", "low", "close"]
    required_regimes = [
        "timestamp",
        "market_regime",
        "volatility_state",
    ]

    for column in required_candles:
        if column not in candles.columns:
            raise ValueError(f"Missing candle column: {column}")

    for column in required_regimes:
        if column not in regimes.columns:
            raise ValueError(f"Missing regime column: {column}")

    if candles["timestamp"].duplicated().any():
        raise ValueError("Duplicate candle timestamps found.")

    if regimes["timestamp"].duplicated().any():
        raise ValueError("Duplicate regime timestamps found.")

    # Calculate the highest high and lowest low
    # across the NEXT 24 completed candles.
    future_high = (
        candles["high"]
        .shift(-1)[::-1]
        .rolling(HORIZON, min_periods=HORIZON)
        .max()[::-1]
    )

    future_low = (
        candles["low"]
        .shift(-1)[::-1]
        .rolling(HORIZON, min_periods=HORIZON)
        .min()[::-1]
    )

    candles["future_high_24h"] = future_high
    candles["future_low_24h"] = future_low

    candles = candles.dropna(
        subset=["future_high_24h", "future_low_24h"]
    ).copy()

    candles = candles.merge(
        regimes[
            [
                "timestamp",
                "market_regime",
                "volatility_state",
            ]
        ],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    if candles.empty:
        raise ValueError("No matching candle and regime records.")

    if (candles["close"] <= 0).any():
        raise ValueError("Invalid closing prices found.")

    # Long position excursion percentages.
    candles["long_mfe"] = (
        (candles["future_high_24h"] / candles["close"]) - 1
    ) * 100

    candles["long_mae"] = (
        (candles["future_low_24h"] / candles["close"]) - 1
    ) * 100

    # Short position excursion percentages.
    candles["short_mfe"] = (
        (1 - candles["future_low_24h"] / candles["close"])
    ) * 100

    candles["short_mae"] = (
        (1 - candles["future_high_24h"] / candles["close"])
    ) * 100

    excursion_columns = [
        "long_mfe",
        "long_mae",
        "short_mfe",
        "short_mae",
    ]

    if not np.isfinite(candles[excursion_columns]).all().all():
        raise ValueError("Invalid excursion values found.")

    results = (
        candles.groupby(
            ["market_regime", "volatility_state"],
            observed=True,
        )
        .agg(
            observations=("close", "count"),
            long_mfe_median=("long_mfe", "median"),
            long_mfe_75th=("long_mfe", lambda x: x.quantile(0.75)),
            long_mae_median=("long_mae", "median"),
            long_mae_25th=("long_mae", lambda x: x.quantile(0.25)),
            short_mfe_median=("short_mfe", "median"),
            short_mfe_75th=("short_mfe", lambda x: x.quantile(0.75)),
            short_mae_median=("short_mae", "median"),
            short_mae_25th=("short_mae", lambda x: x.quantile(0.25)),
        )
        .reset_index()
        .round(4)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\n24-HOUR EXCURSION RESULTS")
    print(results.to_string(index=False))

    print("\nTOTAL OBSERVATIONS:", len(candles))
    print("SAVED TO:", OUTPUT_FILE)


if __name__ == "__main__":
    evaluate_excursions()
