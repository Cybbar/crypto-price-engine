
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/btc_usdt_1h.csv")
OUTPUT_PATH = Path("reports/volatility_model_comparison.csv")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

# Historical hourly returns
df["hourly_return"] = np.log(
    df["close"] / df["close"].shift(1)
)

# Historical 24-hour realized volatility
df["volatility_24h"] = (
    df["hourly_return"].rolling(24).std() * np.sqrt(24)
)

# Efficiency ratio
price_change = (
    df["close"] - df["close"].shift(24)
).abs()

total_movement = (
    df["close"].diff().abs().rolling(24).sum()
)

df["efficiency_ratio"] = price_change / total_movement

# Actual future 12-hour realized volatility
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

    # Purge the final 12 training rows so their future
    # volatility targets cannot overlap the test period.
    train = df.iloc[:test_start - 12].copy()
    test = df.iloc[test_start:test_end].copy()

    # Thresholds learned from training data only
    vol_threshold = train["volatility_24h"].quantile(0.80)
    trend_threshold = train["efficiency_ratio"].quantile(0.60)

    def assign_regime(data):
        high_vol = data["volatility_24h"] >= vol_threshold
        trending = data["efficiency_ratio"] >= trend_threshold

        return np.select(
            [high_vol, trending],
            ["high_volatility", "trending"],
            default="ranging"
        )

    train["regime"] = assign_regime(train)
    test["regime"] = assign_regime(test)

    # Model 1: historical volatility persistence.
    # Convert 24-hour volatility to a 12-hour scale.
    test["persistence_prediction"] = (
        test["volatility_24h"] * np.sqrt(12 / 24)
    )

    # Model 2: unconditional training average
    historical_mean = train["future_volatility_12h"].mean()

    test["historical_mean_prediction"] = historical_mean

    # Model 3: training average for each market regime
    regime_means = (
        train.groupby("regime")["future_volatility_12h"].mean()
    )

    test["regime_prediction"] = (
        test["regime"].map(regime_means)
    )

    # Fallback for a regime missing from training
    test["regime_prediction"] = (
        test["regime_prediction"].fillna(historical_mean)
    )

    actual = test["future_volatility_12h"]

    predictions = {
        "persistence": test["persistence_prediction"],
        "historical_mean": test["historical_mean_prediction"],
        "regime_model": test["regime_prediction"]
    }

    print(f"\n{'=' * 65}")
    print(f"FOLD {fold + 1}")
    print(f"{'=' * 65}")
    print(f"Training observations: {len(train)}")
    print(f"Testing observations: {len(test)}")

    for model_name, prediction in predictions.items():

        mae = np.mean(np.abs(actual - prediction))
        rmse = np.sqrt(np.mean((actual - prediction) ** 2))

        print(
            f"{model_name:18s} "
            f"MAE: {mae * 100:.4f}% | "
            f"RMSE: {rmse * 100:.4f}%"
        )

        results.append({
            "fold": fold + 1,
            "model": model_name,
            "mae": mae,
            "rmse": rmse,
            "test_observations": len(test)
        })

results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

# Overall fold averages
summary = (
    results_df.groupby("model")
    .agg(
        average_mae=("mae", "mean"),
        average_rmse=("rmse", "mean")
    )
    .sort_values("average_mae")
)

print("\n" + "=" * 65)
print("OVERALL MODEL COMPARISON")
print("=" * 65)

for model_name, row in summary.iterrows():
    print(
        f"{model_name:18s} "
        f"Average MAE: {row['average_mae'] * 100:.4f}% | "
        f"Average RMSE: {row['average_rmse'] * 100:.4f}%"
    )

print("\nResults saved to:", OUTPUT_PATH)
