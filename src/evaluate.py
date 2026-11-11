
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "btc_features.csv"

FEATURES = [
    "return_1h",
    "return_24h",
    "volatility_24h",
    "volume_ratio_24h",
    "range_ratio",
    "trend_24h",
]

# Focus on short-term prediction first.
TARGET = "target_return_1h"
HORIZON = 1


def print_metrics(label, actual, predictions):
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    # Ignore exactly zero predictions when measuring directional accuracy.
    nonzero = predictions != 0
    if nonzero.any():
        direction_accuracy = np.mean(
            np.sign(actual[nonzero]) == np.sign(predictions[nonzero])
        )
    else:
        direction_accuracy = np.nan

    print(f"\n{label}")
    print(f"  MAE:                {mae:.6f}")
    print(f"  RMSE:               {rmse:.6f}")
    print(f"  Direction accuracy: {direction_accuracy:.2%}"
          if not np.isnan(direction_accuracy)
          else "  Direction accuracy: N/A")

    return mae, rmse


def main():
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = ["timestamp", TARGET, *FEATURES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).reset_index(drop=True)

    split = int(len(df) * 0.8)

    # Purge the final training label(s) that overlap the test boundary.
    train = df.iloc[:split - HORIZON].copy()
    test = df.iloc[split:].copy()

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    # Baselines
    zero_predictions = np.zeros(len(test))
    mean_predictions = np.full(len(test), y_train.mean())

    # Predict the most common training direction.
    majority_direction = 1 if (y_train > 0).mean() >= 0.5 else -1
    majority_predictions = np.full(
        len(test),
        majority_direction * max(abs(y_train.mean()), 1e-12),
    )

    # Random Forest
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    rf_predictions = model.predict(X_test)

    print("=" * 60)
    print("BTC/USDT — ONE-HOUR OUT-OF-SAMPLE EVALUATION")
    print(f"Training rows: {len(train):,}")
    print(f"Testing rows:  {len(test):,}")
    print(f"Test period:   {test['timestamp'].iloc[0]} to "
          f"{test['timestamp'].iloc[-1]}")
    print(f"Actual positive returns: {(y_test > 0).mean():.2%}")
    print(f"Training positive returns: {(y_train > 0).mean():.2%}")

    results = {}
    for label, predictions in [
        ("Zero baseline", zero_predictions),
        ("Historical mean baseline", mean_predictions),
        ("Majority-direction baseline", majority_predictions),
        ("Random Forest", rf_predictions),
    ]:
        results[label] = print_metrics(label, y_test, predictions)

    zero_mae = results["Zero baseline"][0]
    rf_mae = results["Random Forest"][0]
    improvement = (zero_mae - rf_mae) / zero_mae * 100

    print("\n" + "=" * 60)
    print(f"RF MAE improvement vs zero baseline: {improvement:.2f}%")

    # Correlation between predicted and actual returns.
    correlation = np.corrcoef(rf_predictions, y_test)[0, 1]
    print(f"Prediction/actual correlation: {correlation:.4f}")

    # Examine prediction strength without choosing a trading threshold.
    diagnostic = pd.DataFrame({
        "prediction": rf_predictions,
        "actual": y_test.to_numpy(),
    })

    diagnostic["prediction_bucket"] = pd.qcut(
        diagnostic["prediction"],
        q=5,
        labels=False,
        duplicates="drop",
    )

    print("\nActual returns by predicted-return bucket:")
    bucket_results = diagnostic.groupby("prediction_bucket").agg(
        count=("actual", "size"),
        mean_predicted=("prediction", "mean"),
        mean_actual=("actual", "mean"),
        positive_actual=("actual", lambda x: (x > 0).mean()),
    )

    print(bucket_results.to_string(float_format=lambda x: f"{x:.6f}"))

    # Feature importance is descriptive, not proof of causation.
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    print("\nRandom Forest feature importance:")
    print(importance.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nNo trading profitability is inferred from these metrics.")


if __name__ == "__main__":
    main()
