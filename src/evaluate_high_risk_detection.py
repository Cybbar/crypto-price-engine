import numpy as np
import pandas as pd

DATA_PATH = "data/processed/btc_features.csv"

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Recreate the features and target used in the earlier comparison.
df["hourly_return"] = np.log(df["close"] / df["close"].shift(1))

df["volatility_24h"] = (
    df["hourly_return"]
    .rolling(24)
    .std() * np.sqrt(24)
)

df["efficiency_ratio"] = (
    (df["close"] - df["close"].shift(24)).abs()
    / df["close"].diff().abs().rolling(24).sum()
)

future_returns = df["hourly_return"].shift(-1)

df["future_volatility_12h"] = (
    future_returns.iloc[::-1]
    .rolling(12)
    .std()
    .iloc[::-1] * np.sqrt(12)
)

df = df.dropna().reset_index(drop=True)

folds = 5
initial_train_size = int(len(df) * 0.70)
test_size = (len(df) - initial_train_size) // folds

all_results = []


def calculate_metrics(actual, predicted):
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    false_alarm_rate = fp / (fp + tn) if fp + tn else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


for fold in range(folds):
    print(f"\n{'=' * 55}")
    print(f"FOLD {fold + 1}")
    print("=" * 55)

    test_start = initial_train_size + fold * test_size
    test_end = (
        initial_train_size + (fold + 1) * test_size
        if fold < folds - 1 else len(df)
    )

    train = df.iloc[:test_start - 12].copy()
    test = df.iloc[test_start:test_end].copy()

    # Threshold learned from training data only.
    risk_threshold = train["future_volatility_12h"].quantile(0.80)

    # Learn regime thresholds from training data only.
    vol_threshold = train["volatility_24h"].quantile(0.80)
    trend_threshold = train["efficiency_ratio"].quantile(0.60)

    def assign_regime(data):
        conditions = [
            data["volatility_24h"] >= vol_threshold,
            data["efficiency_ratio"] >= trend_threshold,
        ]

        return np.select(
            conditions,
            ["high_volatility", "trending"],
            default="ranging"
        )

    train["regime"] = assign_regime(train)
    test["regime"] = assign_regime(test)

    # Persistence forecast.
    persistence_prediction = (
        test["volatility_24h"] * np.sqrt(12 / 24)
    )

    # Regime-based forecast.
    regime_means = train.groupby("regime")[
        "future_volatility_12h"
    ].mean()

    regime_prediction = test["regime"].map(regime_means)

    # Actual high-risk periods.
    actual_high_risk = (
        test["future_volatility_12h"] >= risk_threshold
    ).astype(int)

    predictions = {
        "persistence": (
            persistence_prediction >= risk_threshold
        ).astype(int),
        "regime_model": (
            regime_prediction >= risk_threshold
        ).astype(int),
    }

    for model_name, prediction in predictions.items():
        metrics = calculate_metrics(actual_high_risk, prediction)

        print(f"\n{model_name}")
        print(f"Precision:         {metrics['precision']:.3f}")
        print(f"Recall:            {metrics['recall']:.3f}")
        print(f"F1 score:          {metrics['f1']:.3f}")
        print(f"False alarm rate:  {metrics['false_alarm_rate']:.3f}")
        print(
            f"TP: {metrics['TP']} | FP: {metrics['FP']} | "
            f"FN: {metrics['FN']} | TN: {metrics['TN']}"
        )

        all_results.append({
            "fold": fold + 1,
            "model": model_name,
            **metrics,
        })


results = pd.DataFrame(all_results)

print("\n" + "=" * 55)
print("AVERAGE RESULTS ACROSS FOLDS")
print("=" * 55)

summary = results.groupby("model")[
    ["precision", "recall", "f1", "false_alarm_rate"]
].mean()

print(summary.round(3))

results.to_csv(
    "reports/high_risk_detection_results.csv",
    index=False
)

print("\nSaved to reports/high_risk_detection_results.csv")