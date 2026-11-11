
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "btc_features.csv"

FEATURE = "range_ratio"
TARGET = "target_return_1h"
N_SPLITS = 5

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[FEATURE, TARGET]).reset_index(drop=True)

# Keep the earliest 50% for initial history.
# Divide the remaining 50% into five sequential test periods.
initial_train_size = len(df) // 2
remaining = len(df) - initial_train_size
test_size = remaining // N_SPLITS

print("WALK-FORWARD HOURLY RANGE TEST")
print("=" * 72)
print("Signal: range_ratio above/below the training-period median")
print("Target: next-hour return")
print("Thresholds are recalculated from each training period.\n")

all_results = []

for fold in range(N_SPLITS):
    train_end = initial_train_size + fold * test_size
    test_end = (
        initial_train_size + (fold + 1) * test_size
        if fold < N_SPLITS - 1
        else len(df)
    )

    train = df.iloc[:train_end]
    test = df.iloc[train_end:test_end].copy()

    # Learn the threshold from historical data only.
    threshold = train[FEATURE].median()

    test["group"] = np.where(
        test[FEATURE] > threshold,
        "High range",
        "Low range",
    )

    high = test.loc[
        test["group"] == "High range", TARGET
    ]
    low = test.loc[
        test["group"] == "Low range", TARGET
    ]

    high_mean = high.mean()
    low_mean = low.mean()
    difference = high_mean - low_mean

    all_results.append({
        "fold": fold + 1,
        "high_mean": high_mean,
        "low_mean": low_mean,
        "difference": difference,
        "high_count": len(high),
        "low_count": len(low),
    })

    print(f"Fold {fold + 1}")
    print(f"  Test period: {test['timestamp'].iloc[0]} "
          f"to {test['timestamp'].iloc[-1]}")
    print(f"  Training rows: {len(train):,}")
    print(f"  Range threshold: {threshold:.6f}")
    print(f"  High-range mean next return: {high_mean:.6f}")
    print(f"  Low-range mean next return:  {low_mean:.6f}")
    print(f"  Difference (high - low):     {difference:.6f}")
    print()

results = pd.DataFrame(all_results)

print("=" * 72)
print("SUMMARY")
print(f"Folds with positive high-minus-low difference: "
      f"{(results['difference'] > 0).sum()}/{N_SPLITS}")
print(f"Average high-range return: {results['high_mean'].mean():.6f}")
print(f"Average low-range return:  {results['low_mean'].mean():.6f}")
print(f"Average difference:        {results['difference'].mean():.6f}")
print("\nThis tests predictive consistency, not trading profitability.")
