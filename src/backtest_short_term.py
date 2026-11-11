
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "btc_features.csv"

# Backtest assumptions — configurable, not guaranteed exchange costs
FEE_RATE = 0.001       # 0.10% per trade side
SLIPPAGE_RATE = 0.0005 # 0.05% per trade side


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = [
        "timestamp", "open", "close", "return_1h",
        "volatility_24h", "trend_24h"
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            "Check data/processed/btc_features.csv."
        )

    return df


def run_backtest(df, strategy_name, signal, cost_rate):
    """
    Signal from candle t is executed at candle t+1 open.
    The position earns the open-to-close return of that candle.
    """
    df = df.copy()

    # Signal at t becomes the position for candle t+1.
    df["position"] = signal.shift(1).fillna(0)

    # Return earned during the current candle.
    df["market_return"] = df["close"] / df["open"] - 1

    # Charge costs on entries and exits.
    previous_position = df["position"].shift(1).fillna(0)
    turnover = (df["position"] - previous_position).abs()

    df["strategy_return"] = (
        df["position"] * df["market_return"]
        - turnover * cost_rate
    )

    # Drop the first row because no prior signal exists.
    results = df.iloc[1:].copy()
    returns = results["strategy_return"]

    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1
    max_drawdown = (equity / equity.cummax() - 1).min()

    active = results["position"] == 1
    buy_hold_return = (
        (1 + results["market_return"]).prod() - 1
    )

    print(f"\n--- {strategy_name} ---")
    print(f"Test candles: {len(results):,}")
    print(f"Market exposure: {active.mean():.1%}")
    print(f"Position changes: {int(turnover.iloc[1:].sum())}")
    print(f"Total return after estimated costs: {total_return:.2%}")
    print(f"Buy-and-hold return over same candles: {buy_hold_return:.2%}")
    print(f"Maximum drawdown: {max_drawdown:.2%}")

    if active.any():
        win_rate = (
            results.loc[active, "market_return"] > 0
        ).mean()
        print(f"Profitable holding hours: {win_rate:.2%}")

    return results



def main():
    df = load_data()

    # Use the latest 20% as a simple chronological test period.
    # Strategy thresholds are fixed in advance, not fitted to this period.
    split = int(len(df) * 0.8)
    test = df.iloc[split:].copy()

    # Strategy 1: momentum — hold BTC when recent trend is positive.
    momentum_signal = (test["trend_24h"] > 0).astype(float)

    # Strategy 2: mean reversion — hold BTC when price is below
    # its recent average. This is a hypothesis, not a proven edge.
    reversion_signal = (test["trend_24h"] < 0).astype(float)

    # Strategy 3: buy-and-hold reference.
    buy_hold_signal = pd.Series(1.0, index=test.index)

    estimated_cost = FEE_RATE + SLIPPAGE_RATE

    print(f"Test period: {test['timestamp'].iloc[0]} "
          f"to {test['timestamp'].iloc[-1]}")
    print(f"Estimated cost per position change: "
          f"{estimated_cost:.2%}")

    run_backtest(
        test, "Momentum", momentum_signal, estimated_cost
    )
    run_backtest(
        test, "Mean reversion", reversion_signal, estimated_cost
    )
    run_backtest(
        test, "Buy and hold", buy_hold_signal, estimated_cost
    )


if __name__ == "__main__":
    main()
