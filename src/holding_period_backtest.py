
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "btc_usdt_1h.csv"
OUTPUT = ROOT / "reports" / "holding_period_results.csv"

# Illustrative assumptions. Replace with your actual trading costs later.
COST_PER_SIDE = 0.0015
HOLDING_PERIODS = [4, 24]
INITIAL_TRAIN_FRACTION = 0.50
N_FOLDS = 5

# A signal must exceed this estimated move to enter.
# This is a research threshold, not a recommended trading setting.
NO_TRADE_THRESHOLD = 0.002


def load_data():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.reset_index(drop=True)

    required = ["timestamp", "open", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).reset_index(drop=True)

    if (df[["open", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive.")

    return df


def build_signals(train, test, horizon):
    # Past close-to-close momentum, available at the candle close.
    train_momentum = np.log(
        train["close"] / train["close"].shift(horizon)
    ).dropna()

    test_momentum = np.log(
        test["close"] / test["close"].shift(horizon)
    )

    # Require stronger historical momentum than the fixed minimum.
    # Use training data only to set the threshold.
    historical_abs = train_momentum.abs().dropna()
    threshold = max(
        NO_TRADE_THRESHOLD,
        historical_abs.quantile(0.60),
    )

    signal = pd.Series(0.0, index=test.index)
    signal.loc[test_momentum > threshold] = 1.0
    signal.loc[test_momentum < -threshold] = 0.0

    return signal, threshold


def evaluate(test, signal, name):
    test = test.reset_index(drop=True).copy()
    signal = signal.reset_index(drop=True)

    # Signals calculated at close t are executed at open t+1.
    position = signal.shift(1).fillna(0.0)

    opens = test["open"].to_numpy()
    closes = test["close"].to_numpy()

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    exposure_count = 0
    turnover_count = 0

    for i in range(1, len(test)):
        previous_position = position.iloc[i - 1]
        current_position = position.iloc[i]

        # Approximate trade costs at the current candle's open.
        turnover = abs(current_position - previous_position)

        if turnover:
            equity *= max(0.0, 1.0 - turnover * COST_PER_SIDE)
            turnover_count += int(turnover)

        # A position held overnight across hourly candles earns the
        # previous close -> current open gap and current open -> close.
        if previous_position:
            equity *= max(0.0, opens[i] / closes[i - 1])

        if current_position:
            equity *= max(0.0, closes[i] / opens[i])
            exposure_count += 1

        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)

    # Close any remaining position at the end of the test.
    if position.iloc[-1] > 0:
        equity *= 1.0 - COST_PER_SIDE

    return {
        "strategy": name,
        "net_return": equity - 1.0,
        "max_drawdown": max_drawdown,
        "exposure": exposure_count / max(1, len(test) - 1),
        "position_changes": turnover_count,
    }


def main():
    df = load_data()

    initial_train = int(len(df) * INITIAL_TRAIN_FRACTION)
    remaining = len(df) - initial_train
    fold_size = remaining // N_FOLDS
    results = []

    print("BTC HOLDING-PERIOD EXPERIMENT")
    print("=" * 75)
    print(f"Estimated cost per side: {COST_PER_SIDE:.2%}")
    print(f"No-trade threshold floor: {NO_TRADE_THRESHOLD:.2%}")

    for fold in range(N_FOLDS):
        train_end = initial_train + fold * fold_size
        test_end = (
            initial_train + (fold + 1) * fold_size
            if fold < N_FOLDS - 1
            else len(df)
        )

        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end:test_end].copy()

        print("\n" + "-" * 75)
        print(f"FOLD {fold + 1}")
        print(
            f"{test['timestamp'].iloc[0]} to "
            f"{test['timestamp'].iloc[-1]}"
        )

        # Buy and hold is a benchmark; cash is the zero-return baseline.
        for name, signal in [
            ("Cash", pd.Series(0.0, index=test.index)),
            ("Buy and hold", pd.Series(1.0, index=test.index)),
        ]:
            result = evaluate(test, signal, name)
            result["fold"] = fold + 1
            results.append(result)
            print(
                f"{name:18} | Return {result['net_return']:8.2%} | "
                f"Drawdown {result['max_drawdown']:8.2%}"
            )

        for horizon in HOLDING_PERIODS:
            signal, threshold = build_signals(train, test, horizon)
            name = f"{horizon}h momentum"
            result = evaluate(test, signal, name)
            result["fold"] = fold + 1
            result["threshold"] = threshold
            results.append(result)

            print(
                f"{name:18} | Return {result['net_return']:8.2%} | "
                f"Drawdown {result['max_drawdown']:8.2%} | "
                f"Exposure {result['exposure']:6.1%} | "
                f"Changes {result['position_changes']:4d} | "
                f"Threshold {threshold:.3%}"
            )

    results_df = pd.DataFrame(results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT, index=False)

    print("\n" + "=" * 75)
    print("AVERAGE PER-FOLD SUMMARY")
    summary = results_df.groupby("strategy").agg(
        average_return=("net_return", "mean"),
        median_return=("net_return", "median"),
        average_drawdown=("max_drawdown", "mean"),
        average_exposure=("exposure", "mean"),
        total_changes=("position_changes", "sum"),
    )
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved results to: {OUTPUT}")


if __name__ == "__main__":
    main()
