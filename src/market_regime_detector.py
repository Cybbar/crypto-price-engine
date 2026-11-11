
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "btc_usdt_1h.csv"
OUTPUT = ROOT / "reports" / "market_regime_results.csv"

N_FOLDS = 5
INITIAL_TRAIN_FRACTION = 0.70
LOOKBACK = 24
FUTURE_HORIZON = 12


def build_features(df):
    df = df.copy()

    # Hourly log returns
    df["hourly_return"] = np.log(
        df["close"] / df["close"].shift(1)
    )

    # Realized volatility over the previous 24 hours
    df["volatility_24h"] = (
        df["hourly_return"].rolling(LOOKBACK).std()
        * np.sqrt(LOOKBACK)
    )

    # Directional efficiency:
    # Near 1 = relatively consistent movement.
    # Near 0 = movement with little net direction.
    net_movement = (
        df["close"] - df["close"].shift(LOOKBACK)
    ).abs()

    total_movement = (
        df["close"].diff().abs().rolling(LOOKBACK).sum()
    )

    df["efficiency_ratio"] = (
        net_movement / total_movement.replace(0, np.nan)
    )

    # Future return is for evaluation only, never classification.
    df["future_12h_return"] = np.log(
        df["close"].shift(-FUTURE_HORIZON) / df["close"]
    )

    return df


def evaluate_regimes(test, volatility_threshold, trend_threshold):
    test = test.copy()

    # High volatility takes priority over the other regimes.
    conditions = [
        test["volatility_24h"] >= volatility_threshold,
        test["efficiency_ratio"] >= trend_threshold,
    ]

    choices = ["high_volatility", "trending"]

    test["regime"] = np.select(
        conditions,
        choices,
        default="ranging",
    )

    results = []

    for regime in ["trending", "ranging", "high_volatility"]:
        sample = test[
            (test["regime"] == regime)
            & test["future_12h_return"].notna()
        ]

        if len(sample) == 0:
            continue

        future = sample["future_12h_return"]

        results.append({
            "regime": regime,
            "observations": len(sample),
            "mean_future_return_pct": future.mean() * 100,
            "median_future_return_pct": future.median() * 100,
            "positive_return_pct": (future > 0).mean() * 100,
            "mean_future_volatility_pct": (
                sample["volatility_24h"].mean() * 100
            ),
        })

    return results


def main():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    df = build_features(df)

    initial_train = int(len(df) * INITIAL_TRAIN_FRACTION)
    remaining = len(df) - initial_train
    fold_size = remaining // N_FOLDS

    all_results = []

    for fold in range(N_FOLDS):
        start = initial_train + fold * fold_size
        end = (
            initial_train + (fold + 1) * fold_size
            if fold < N_FOLDS - 1
            else len(df)
        )

        train = df.iloc[:start].copy()
        test = df.iloc[start:end].copy()

        # Thresholds are estimated using training data only.
        volatility_threshold = train[
            "volatility_24h"
        ].quantile(0.80)

        trend_threshold = train[
            "efficiency_ratio"
        ].quantile(0.60)

        test_results = evaluate_regimes(
            test,
            volatility_threshold,
            trend_threshold,
        )

        print(f"\n{'=' * 65}")
        print(f"FOLD {fold + 1}")
        print(
            f"Test period: {test['timestamp'].iloc[0]} "
            f"to {test['timestamp'].iloc[-1]}"
        )
        print(
            f"High-volatility threshold: "
            f"{volatility_threshold:.4%}"
        )
        print(
            f"Trending threshold: {trend_threshold:.4f}"
        )

        for result in test_results:
            result["fold"] = fold + 1
            result["volatility_threshold"] = volatility_threshold
            result["trend_threshold"] = trend_threshold
            all_results.append(result)

            print(
                f"{result['regime']:18s} | "
                f"Count: {result['observations']:5d} | "
                f"Mean future return: "
                f"{result['mean_future_return_pct']:8.4f}% | "
                f"Median: "
                f"{result['median_future_return_pct']:8.4f}% | "
                f"Positive: "
                f"{result['positive_return_pct']:6.2f}%"
            )

    output = pd.DataFrame(all_results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)

    print(f"\nSaved results to: {OUTPUT}")
    print("\nThis is a market-regime diagnostic, not a trading strategy.")
    print("Future returns are used only to evaluate the classifications.")


if __name__ == "__main__":
    main()
