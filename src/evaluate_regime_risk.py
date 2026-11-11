
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/btc_usdt_1h.csv")
OUTPUT_PATH = Path("reports/regime_risk_results.csv")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

# Calculate hourly log returns
df["hourly_return"] = np.log(df["close"] / df["close"].shift(1))

# Future 12-hour absolute return
df["future_12h_return"] = (
    np.log(df["close"].shift(-12) / df["close"])
)

df["future_abs_12h_return"] = df["future_12h_return"].abs()

# Future 12-hour realized volatility.
# Uses returns from the next hour through the next 12 hours.
future_returns = df["hourly_return"].shift(-1)

df["future_volatility_12h"] = (
    future_returns.iloc[::-1]
    .rolling(12)
    .std()
    .iloc[::-1]
    * np.sqrt(12)
)

# Recreate the regime features
df["volatility_24h"] = (
    df["hourly_return"]
    .rolling(24)
    .std()
    * np.sqrt(24)
)

price_change = (df["close"] - df["close"].shift(24)).abs()
total_movement = df["close"].diff().abs().rolling(24).sum()

df["efficiency_ratio"] = price_change / total_movement

df = df.dropna().reset_index(drop=True)

# Same five chronological walk-forward folds
initial_train_size = int(len(df) * 0.70)
remaining = len(df) - initial_train_size
fold_size = remaining // 5

results = []

for fold in range(5):
    test_start = initial_train_size + fold * fold_size

    test_end = (
        initial_train_size + (fold + 1) * fold_size
        if fold < 4
        else len(df)
    )

    train = df.iloc[:test_start]
    # Use non-overlapping 12-hour observations
    test = df.iloc[test_start:test_end:12].copy()

    # Thresholds learned from training data only
    vol_threshold = train["volatility_24h"].quantile(0.80)
    trend_threshold = train["efficiency_ratio"].quantile(0.60)

    high_vol = test["volatility_24h"] >= vol_threshold
    trending = test["efficiency_ratio"] >= trend_threshold

    test["regime"] = np.select(
        [high_vol, trending],
        ["high_volatility", "trending"],
        default="ranging"
    )

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold + 1}")
    print(f"{'=' * 60}")

    for regime in ["trending", "ranging", "high_volatility"]:
        subset = test[test["regime"] == regime]

        if len(subset) == 0:
            continue

        avg_abs_return = subset["future_abs_12h_return"].mean() * 100
        avg_future_vol = subset["future_volatility_12h"].mean() * 100

        print(f"\n{regime}")
        print(f"Observations: {len(subset)}")
        print(f"Average absolute 12h return: {avg_abs_return:.4f}%")
        print(f"Average future 12h volatility: {avg_future_vol:.4f}%")

        results.append({
            "fold": fold + 1,
            "regime": regime,
            "observations": len(subset),
            "avg_abs_12h_return_pct": avg_abs_return,
            "avg_future_volatility_12h_pct": avg_future_vol
        })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nResults saved to: {OUTPUT_PATH}")
