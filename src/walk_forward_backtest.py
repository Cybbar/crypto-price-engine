
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "btc_features.csv"
OUTPUT = ROOT / "reports" / "walk_forward_results.csv"

N_FOLDS = 5
INITIAL_TRAIN_FRACTION = 0.50

# Illustrative cost per side: 0.10% fee + 0.05% slippage.
COST_PER_SIDE = 0.0015


def load_data():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = ["timestamp", "open", "close", "return_1h", "range_ratio"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).reset_index(drop=True)

    if len(df) < 1000:
        raise ValueError("Not enough usable data.")

    if (df[["open", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive.")

    return df


def build_signals(train, test):
    # Thresholds are learned using training data only.
    momentum_threshold = train["return_1h"].median()
    range_threshold = train["range_ratio"].median()

    return {
        "Momentum": (
            test["return_1h"] > momentum_threshold
        ).astype(float),

        "Mean reversion": (
            test["return_1h"] < momentum_threshold
        ).astype(float),

        "High range": (
            test["range_ratio"] > range_threshold
        ).astype(float),

        "Cash": pd.Series(0.0, index=test.index),

        "Buy and hold": pd.Series(1.0, index=test.index),
    }


def backtest(test, signal, name):
    """
    Signal at candle t's close determines exposure from candle t+1's open.

    Equity is marked at each candle's close. Entry/exit costs are applied
    at the candle open when exposure changes.
    """
    test = test.copy().reset_index(drop=True)
    signal = signal.reset_index(drop=True).astype(float)

    # A signal from the previous completed candle is used at this open.
    position = signal.shift(1).fillna(0.0)

    # Use close-to-next-open movement plus open-to-close movement.
    # Together these account for the full close-to-close price movement
    # while the strategy is holding a position.
    open_price = test["open"]
    close_price = test["close"]

    # First test candle has no prior test-period signal, so it starts flat.
    previous_close = close_price.shift(1)
    previous_close.iloc[0] = open_price.iloc[0]

    gap_return = open_price / previous_close - 1.0
    intrabar_return = close_price / open_price - 1.0

    # Exposure entering the candle earns both gap and intrabar returns.
    # The first row begins at its open, so it has no preceding gap return.
    gap_return.iloc[0] = 0.0

    previous_position = position.shift(1).fillna(0.0)
    turnover = (position - previous_position).abs()

    # Apply trading costs at the open, then calculate market movement.
    # Multiplication keeps the accounting consistent with compounded equity.
    equity = 1.0
    equity_curve = []

    for i in range(len(test)):
        if turnover.iloc[i] > 0:
            equity *= max(
                0.0,
                1.0 - turnover.iloc[i] * COST_PER_SIDE
            )

        if position.iloc[i] > 0:
            equity *= max(0.0, 1.0 + gap_return.iloc[i])
            equity *= max(0.0, 1.0 + intrabar_return.iloc[i])

        equity_curve.append(equity)

    curve = pd.Series(equity_curve, dtype=float)
    running_peak = curve.cummax()
    drawdown = curve / running_peak - 1.0

    # Close any remaining position at the end of the test period.
    if position.iloc[-1] > 0:
        equity *= 1.0 - COST_PER_SIDE

    return {
        "strategy": name,
        "net_return": equity - 1.0,
        "max_drawdown": drawdown.min(),
        "exposure": position.mean(),
        "position_changes": int(turnover.sum()),
    }


def main():
    df = load_data()
    diagnose_first_fold(df)
    initial_train = int(len(df) * INITIAL_TRAIN_FRACTION)
    fold_size = (len(df) - initial_train) // N_FOLDS

    results = []

    print("BTC/USDT WALK-FORWARD BACKTEST")
    print("=" * 78)
    print(f"Estimated transaction cost per side: {COST_PER_SIDE:.2%}")
    print(f"Initial training rows: {initial_train:,}")
    print(f"Number of test folds: {N_FOLDS}")

    for fold in range(N_FOLDS):
        train_end = initial_train + fold * fold_size
        test_end = (
            initial_train + (fold + 1) * fold_size
            if fold < N_FOLDS - 1
            else len(df)
        )

        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end:test_end].copy()

        signals = build_signals(train, test)

        print("\n" + "-" * 78)
        print(f"FOLD {fold + 1}")
        print(
            f"Test period: {test['timestamp'].iloc[0]} "
            f"to {test['timestamp'].iloc[-1]}"
        )

        for name, signal in signals.items():
            result = backtest(test, signal, name)
            result["fold"] = fold + 1
            result["start"] = test["timestamp"].iloc[0]
            result["end"] = test["timestamp"].iloc[-1]
            results.append(result)

            print(
                f"{name:16} | "
                f"Return: {result['net_return']:8.2%} | "
                f"Max DD: {result['max_drawdown']:8.2%} | "
                f"Exposure: {result['exposure']:6.1%} | "
                f"Changes: {result['position_changes']}"
            )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 78)
    print("AVERAGE PER-FOLD SUMMARY")

    summary = results_df.groupby("strategy").agg(
        average_return=("net_return", "mean"),
        median_return=("net_return", "median"),
        average_max_drawdown=("max_drawdown", "mean"),
        average_exposure=("exposure", "mean"),
    )

    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT, index=False)

    print(f"\nDetailed results saved to: {OUTPUT}")


def diagnose_first_fold(df):
    initial_train = int(len(df) * INITIAL_TRAIN_FRACTION)
    remaining = len(df) - initial_train
    fold_size = remaining // N_FOLDS

    train = df.iloc[:initial_train].copy()
    test = df.iloc[initial_train:initial_train + fold_size].copy()

    signals = build_signals(train, test)

    print("\nBACKTEST DIAGNOSTICS — FIRST FOLD")
    print("=" * 60)

    for name, signal in signals.items():
        position = signal.reset_index(drop=True).shift(1).fillna(0.0)
        turnover = position.diff().abs()
        turnover.iloc[0] = abs(position.iloc[0])

        print(f"\n{name}")
        print(f"Signal mean: {signal.mean():.4f}")
        print(f"Position mean: {position.mean():.4f}")
        print(f"Position changes: {turnover.sum():.0f}")
        print(f"Total estimated costs: {turnover.sum() * COST_PER_SIDE:.2%}")

    market_returns = test["close"] / test["open"] - 1
    print("\nAverage hourly open-to-close return:",
          f"{market_returns.mean():.6%}")
    print("Hourly return standard deviation:",
          f"{market_returns.std():.6%}")

if __name__ == "__main__":
    main()
