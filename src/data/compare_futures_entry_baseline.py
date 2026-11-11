from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR / "data" / "futures" / "processed"
    / "btcusdt_futures_regimes.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "data" / "futures" / "processed"
    / "futures_entry_baseline_comparison.csv"
)

ROUND_TRIP_COST = 0.003
HORIZONS = [1, 4, 12, 24]
ADX_MINIMUM = 25


def compare_baseline():
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    trending = df["market_regime"].eq("trending")
    strong_trend = df["adx_14"] >= ADX_MINIMUM

    signals = {
        "long": (
            trending
            & strong_trend
            & (df["ema_20"] > df["ema_50"])
        ),
        "short": (
            trending
            & strong_trend
            & (df["ema_20"] < df["ema_50"])
        ),
    }

    results = []

    for direction in ["long", "short"]:
        for horizon in HORIZONS:

            entry = df["open"].shift(-1)
            exit_price = df["close"].shift(-horizon)

            if direction == "long":
                gross_return = (exit_price - entry) / entry
            else:
                gross_return = (entry - exit_price) / entry

            net_return = gross_return - ROUND_TRIP_COST

            valid = (
                entry.notna()
                & exit_price.notna()
                & np.isfinite(net_return)
                & df["volatility_state"].notna()
            )

            for volatility in ["normal", "high"]:

                baseline_mask = (
                    valid
                    & df["volatility_state"].eq(volatility)
                )

                signal_mask = baseline_mask & signals[direction]

                for group_name, mask in [
                    ("filtered_signal", signal_mask),
                    ("unfiltered_baseline", baseline_mask),
                ]:
                    values = net_return[mask]

                    if values.empty:
                        continue

                    results.append({
                        "direction": direction,
                        "horizon_hours": horizon,
                        "volatility_state": volatility,
                        "group": group_name,
                        "trades": len(values),
                        "win_rate_pct": values.gt(0).mean() * 100,
                        "mean_net_return_pct": values.mean() * 100,
                        "median_net_return_pct": values.median() * 100,
                    })

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT_FILE, index=False)

    print("\nENTRY SIGNAL VS UNFILTERED BASELINE")
    print(output.round(4).to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    compare_baseline()