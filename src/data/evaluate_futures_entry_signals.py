from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "futures"
    / "processed"
    / "btcusdt_futures_regimes.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "futures"
    / "processed"
    / "futures_entry_signal_evaluation.csv"
)

ROUND_TRIP_COST = 0.003
HORIZONS = [1, 4, 12, 24]
ADX_MINIMUM = 25


def evaluate():
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Entry signals are generated using completed candles.
    trending = df["market_regime"].eq("trending")
    strong_trend = df["adx_14"] >= ADX_MINIMUM

    long_signal = (
        trending
        & strong_trend
        & (df["ema_20"] > df["ema_50"])
    )

    short_signal = (
        trending
        & strong_trend
        & (df["ema_20"] < df["ema_50"])
    )

    results = []

    for direction, signal in [
        ("long", long_signal),
        ("short", short_signal),
    ]:
        for horizon in HORIZONS:

            entry_price = df["open"].shift(-1)
            exit_price = df["close"].shift(-horizon)

            if direction == "long":
                gross_return = (exit_price - entry_price) / entry_price
            else:
                gross_return = (entry_price - exit_price) / entry_price
            net_return = gross_return - ROUND_TRIP_COST

            valid = (
                signal
                & entry_price.notna()
                & exit_price.notna()
                & np.isfinite(net_return)
            )

            sample = pd.DataFrame({
                "net_return": net_return[valid],
                "volatility_state": df.loc[
                    valid, "volatility_state"
                ],
            })

            for volatility, group in sample.groupby(
                "volatility_state"
            ):
                results.append({
                    "direction": direction,
                    "horizon_hours": horizon,
                    "volatility_state": volatility,
                    "trades": len(group),
                    "win_rate_pct": (
                        group["net_return"].gt(0).mean() * 100
                    ),
                    "mean_net_return_pct": (
                        group["net_return"].mean() * 100
                    ),
                    "median_net_return_pct": (
                        group["net_return"].median() * 100
                    ),
                })

    if not results:
        raise ValueError("No valid entry signals were generated.")

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT_FILE, index=False)

    print("\nENTRY SIGNAL EVALUATION")
    print(output.round(4).to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()