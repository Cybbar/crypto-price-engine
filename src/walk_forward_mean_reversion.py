
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "btc_usdt_1h.csv"
OUTPUT = ROOT / "reports" / "corrected_backtest.csv"

FEE_PER_SIDE = 0.0015
HOLD_HOURS = 12
N_FOLDS = 5
INITIAL_TRAIN_FRACTION = 0.70


def calculate_metrics(returns, position):
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    return {
        "return_pct": (equity.iloc[-1] - 1) * 100,
        "max_drawdown_pct": drawdown.min() * 100,
        "exposure_pct": position.mean() * 100,
        "position_changes": int((position.diff().abs().fillna(position.iloc[0])).sum()),
    }


def run_strategy(test, threshold, strategy):
    df = test.copy().reset_index(drop=True)

    # Signal is calculated at each candle close.
    past_return = np.log(df["close"] / df["close"].shift(24))

    if strategy == "mean_reversion":
        desired = (past_return <= -threshold).astype(float)
    elif strategy == "momentum":
        desired = (past_return >= threshold).astype(float)
    elif strategy == "buy_hold":
        desired = pd.Series(1.0, index=df.index)
    elif strategy == "cash":
        desired = pd.Series(0.0, index=df.index)
    else:
        raise ValueError(strategy)

    # Execute decisions at the next candle open.
    desired = desired.shift(1).fillna(0.0)

    # Reassess only every HOLD_HOURS, using the signal available
    # at the close immediately before the new holding period.
    position = pd.Series(0.0, index=df.index)

    for start in range(0, len(df), HOLD_HOURS):
        end = min(start + HOLD_HOURS, len(df))
        position.iloc[start:end] = desired.iloc[start]

    # Open-to-open returns. The final row has no observed next open,
    # so it is excluded from the performance calculation.
    asset_returns = (
        df["open"].shift(-1) / df["open"] - 1
    )

    valid = asset_returns.notna()
    position = position[valid].reset_index(drop=True)
    asset_returns = asset_returns[valid].reset_index(drop=True)

    # Charge for both entry and exit. A final open position is
    # liquidated at the last available open.
    previous = position.shift(1).fillna(0.0)

    turnover = (position - previous).abs()

    # One fee for each actual position change.
    # Include liquidation of any remaining position at the end.
    turnover.iloc[-1] += position.iloc[-1]

    net_returns = (
	position * asset_returns - turnover * FEE_PER_SIDE
    ).clip(lower=-0.999)

    return calculate_metrics(net_returns, position)


def main():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.reset_index(drop=True)

    initial_train = int(len(df) * INITIAL_TRAIN_FRACTION)
    remaining = len(df) - initial_train
    fold_size = remaining // N_FOLDS

    results = []

    for fold in range(N_FOLDS):
        start = initial_train + fold * fold_size
        end = (
            initial_train + (fold + 1) * fold_size
            if fold < N_FOLDS - 1
            else len(df)
        )

        # Include 24 hours of preceding data for signal warm-up.
        test_start = max(0, start - 24)
        train = df.iloc[:start].copy()
        test = df.iloc[test_start:end].copy()

        train_returns = np.log(
            train["close"] / train["close"].shift(24)
        ).dropna()

        threshold = abs(train_returns.quantile(0.20))

        # Exclude warm-up rows from reported test performance.
        test = test.iloc[start - test_start:].reset_index(drop=True)

        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}")
        print(f"Test: {test['timestamp'].iloc[0]} to "
              f"{test['timestamp'].iloc[-1]}")
        print(f"Threshold: {threshold:.4%}")

        for strategy in [
            "mean_reversion",
            "momentum",
            "buy_hold",
            "cash",
        ]:
            metrics = run_strategy(test, threshold, strategy)

            results.append({
                "fold": fold + 1,
                "strategy": strategy,
                "threshold": threshold,
                **metrics,
            })

            print(
                f"{strategy:18s} | "
                f"Return: {metrics['return_pct']:8.2f}% | "
                f"Max DD: {metrics['max_drawdown_pct']:8.2f}% | "
                f"Exposure: {metrics['exposure_pct']:6.2f}% | "
                f"Changes: {metrics['position_changes']}"
            )

    result_df = pd.DataFrame(results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")
    print("\nThis is still a simplified backtest.")
    print("Validate execution assumptions before considering live trading.")


if __name__ == "__main__":
    main()