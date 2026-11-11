
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/btc_usdt_1h.csv")
OUTPUT_PATH = Path("reports/regime_offset_results.csv")

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

# Historical features
df["hourly_return"] = np.log(df["close"] / df["close"].shift(1))

df["volatility_24h"] = (
    df["hourly_return"].rolling(24).std() * np.sqrt(24)
)

price_change = (df["close"] - df["close"].shift(24)).abs()
total_movement = df["close"].diff().abs().rolling(24).sum()

df["efficiency_ratio"] = price_change / total_movement

# Future outcomes: next 12 hours
df["future_12h_return"] = np.log(
    df["close"].shift(-12) / df["close"]
)

df["future_abs_12h_return"] = df["future_12h_return"].abs()

future_returns = df["hourly_return"].shift(-1)

df["future_volatility_12h"] = (
    future_returns.iloc[::-1]
    .rolling(12)
    .std()
    .iloc[::-1]
    * np.sqrt(12)
)

df = df.dropna().reset_index(drop=True)

# Five chronological walk-forward folds
initial_train_size = int(len(df) * 0.70)
remaining = len(df) - initial_train_size
fold_size = remaining // 5

results = []

for fold in range(5):
    test_start = initial_train_size + fold * fold_size
    test_end = (
        initial_train_size + (fold + 1) * fold_size
        if fold < 4 else len(df)
    )

    train = df.iloc[:test_start]
    test = df.iloc[test_start:test_end].copy()

    # Thresholds learned from training only
    vol_threshold = train["volatility_24h"].quantile(0.80)
    trend_threshold = train["efficiency_ratio"].quantile(0.60)

    for offset in range(12):
        sample = test.iloc[offset::12].copy()

        high_vol = sample["volatility_24h"] >= vol_threshold
        trending = sample["efficiency_ratio"] >= trend_threshold

        sample["regime"] = np.select(
            [high_vol, trending],
            ["high_volatility", "trending"],
            default="ranging"
        )

        for regime in ["trending", "ranging", "high_volatility"]:
            subset = sample[sample["regime"] == regime]

            if len(subset) == 0:
                continue

            results.append({
                "fold": fold + 1,
                "offset": offset,
                "regime": regime,
                "observations": len(subset),
                "avg_abs_return_pct": (
                    subset["future_abs_12h_return"].mean() * 100
                ),
                "avg_future_volatility_pct": (
                    subset["future_volatility_12h"].mean() * 100
                )
            })

results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

# Summary across all offsets
summary = (
    results_df.groupby(["fold", "regime"])
    .agg(
        observations=("observations", "sum"),
        avg_abs_return_pct=("avg_abs_return_pct", "mean"),
        avg_future_volatility_pct=(
            "avg_future_volatility_pct", "mean"
        )
    )
    .reset_index()
)

print("\n" + "=" * 65)
print("AVERAGE RESULTS ACROSS ALL 12 OFFSETS")
print("=" * 65)

for fold in range(1, 6):
    print(f"\nFOLD {fold}")

    fold_data = summary[summary["fold"] == fold]

    for _, row in fold_data.iterrows():
        print(
            f"{row['regime']:18s} "
            f"Observations: {int(row['observations']):4d} | "
            f"Future volatility: "
            f"{row['avg_future_volatility_pct']:.4f}% | "
            f"Absolute return: {row['avg_abs_return_pct']:.4f}%"
        )

print("\nResults saved to:", OUTPUT_PATH)
