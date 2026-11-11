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
    / "futures_atr_excursion_evaluation.csv"
)

HORIZON = 24


def evaluate_atr_excursions():
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

    required_candles = [
        "timestamp", "high", "low", "close"
    ]

    required_regimes = [
        "timestamp",
        "market_regime",
        "volatility_state",
        "atr_14",
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

    # Calculate future 24-hour high and low.
    candles["future_high_24h"] = (
        candles["high"]
        .shift(-1)[::-1]
        .rolling(HORIZON, min_periods=HORIZON)
        .max()[::-1]
    )

    candles["future_low_24h"] = (
        candles["low"]
        .shift(-1)[::-1]
        .rolling(HORIZON, min_periods=HORIZON)
        .min()[::-1]
    )

    candles = candles.dropna(
        subset=["future_high_24h", "future_low_24h"]
    ).copy()

    # Attach regime and ATR information at entry time.
    candles = candles.merge(
        regimes[
            [
                "timestamp",
                "market_regime",
                "volatility_state",
                "atr_14",
            ]
        ],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    if candles.empty:
        raise ValueError("No matching candle and regime records.")

    if (
        (candles["close"] <= 0).any()
        or (candles["atr_14"] <= 0).any()
    ):
        raise ValueError("Invalid close or ATR values found.")

    # Normalise excursions using ATR known at entry.
    candles["long_mfe_atr"] = (
        candles["future_high_24h"] - candles["close"]
    ) / candles["atr_14"]

    candles["long_mae_atr"] = (
        candles["future_low_24h"] - candles["close"]
    ) / candles["atr_14"]

    candles["short_mfe_atr"] = (
        candles["close"] - candles["future_low_24h"]
    ) / candles["atr_14"]

    candles["short_mae_atr"] = (
        candles["close"] - candles["future_high_24h"]
    ) / candles["atr_14"]

    excursion_columns = [
        "long_mfe_atr",
        "long_mae_atr",
        "short_mfe_atr",
        "short_mae_atr",
    ]

    if not np.isfinite(
        candles[excursion_columns]
    ).all().all():
        raise ValueError("Invalid ATR excursion values found.")

    # Six-month chronological periods.
    candles["period"] = (
        candles["timestamp"].dt.year.astype(str)
        + "-H"
        + np.where(
            candles["timestamp"].dt.month <= 6,
            "1",
            "2",
        )
    )

    results = (
        candles.groupby(
            [
                "period",
                "market_regime",
                "volatility_state",
            ],
            observed=True,
        )
        .agg(
            observations=("close", "count"),
            long_mfe_median=("long_mfe_atr", "median"),
            long_mfe_75th=(
                "long_mfe_atr",
                lambda x: x.quantile(0.75),
            ),
            long_mae_median=("long_mae_atr", "median"),
            long_mae_25th=(
                "long_mae_atr",
                lambda x: x.quantile(0.25),
            ),
            short_mfe_median=("short_mfe_atr", "median"),
            short_mfe_75th=(
                "short_mfe_atr",
                lambda x: x.quantile(0.75),
            ),
            short_mae_median=("short_mae_atr", "median"),
            short_mae_25th=(
                "short_mae_atr",
                lambda x: x.quantile(0.25),
            ),
        )
        .reset_index()
        .round(4)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nATR-NORMALISED EXCURSION RESULTS")
    print(results.to_string(index=False))

    print("\nTOTAL OBSERVATIONS:", len(candles))
    print("PERIODS:", candles["period"].nunique())
    print("SAVED TO:", OUTPUT_FILE)


if __name__ == "__main__":
    evaluate_atr_excursions()
