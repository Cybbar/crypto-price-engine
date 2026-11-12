from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CANDLE_FILE = ROOT / "data" / "futures" / "raw" / "btcusdt_perpetual_1h.csv"
REGIME_FILE = ROOT / "data" / "futures" / "processed" / "btcusdt_futures_regimes.csv"
OUTPUT_DIR = ROOT / "data" / "futures" / "processed"
TRADES_FILE = OUTPUT_DIR / "futures_breakout_trades.csv"
SUMMARY_FILE = OUTPUT_DIR / "futures_breakout_summary.csv"

# Experimental assumptions: these are not verified Binance costs.
FEE_RATE = 0.001           # 0.10% per side
SLIPPAGE_RATE = 0.0005    # 0.05% per side
ROUND_TRIP_COST_PCT = 200 * (FEE_RATE + SLIPPAGE_RATE)  # 0.30%

LOOKBACK = 24
HORIZON = 24
EXIT_COMBINATIONS = [
    (1.0, 1.5),
    (1.0, 2.0),
    (1.5, 2.0),
    (1.5, 3.0),
    (2.0, 3.0),
    (2.0, 4.0),
]
N_SPLITS = 5
INITIAL_TRAIN_FRACTION = 0.50  # Warm-up history; no model is trained.


def load_data():
    for path in (CANDLE_FILE, REGIME_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    candles = pd.read_csv(CANDLE_FILE)
    regimes = pd.read_csv(REGIME_FILE)

    candle_columns = {"timestamp", "open", "high", "low", "close"}
    regime_columns = {"timestamp", "atr_14"}
    if not candle_columns.issubset(candles.columns):
        raise ValueError(f"Missing candle columns: {sorted(candle_columns - set(candles.columns))}")
    if not regime_columns.issubset(regimes.columns):
        raise ValueError(f"Missing regime columns: {sorted(regime_columns - set(regimes.columns))}")

    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    regimes["timestamp"] = pd.to_datetime(regimes["timestamp"], utc=True)

    if candles["timestamp"].duplicated().any() or regimes["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found in input data.")

    df = candles[list(candle_columns)].merge(
        regimes[["timestamp", "atr_14"]],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("Not enough overlapping candle and ATR data.")

    number_columns = ["open", "high", "low", "close", "atr_14"]
    df[number_columns] = df[number_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[number_columns].to_numpy()).all():
        raise ValueError("Missing or infinite price/ATR data.")
    if (df[number_columns] <= 0).any().any():
        raise ValueError("Prices and ATR must be positive.")
    if ((df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC candles detected.")
    if not df["timestamp"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Merged data has missing hourly candles.")

    return df


def generate_signals(df):
    # Exclude the current candle from the historical range.
    previous_high = df["high"].rolling(LOOKBACK, min_periods=LOOKBACK).max().shift(1)
    previous_low = df["low"].rolling(LOOKBACK, min_periods=LOOKBACK).min().shift(1)

    # Signals only become known when the current candle closes.
    df["long_signal"] = df["close"] > previous_high
    df["short_signal"] = df["close"] < previous_low
    return df


def simulate_trade(future, entry_price, atr, direction, stop_atr, target_atr):
    stop_distance = stop_atr * atr
    target_distance = target_atr * atr

    if direction == "long":
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance
    else:
        stop_price = entry_price + stop_distance
        target_price = entry_price - target_distance

    for offset, candle in enumerate(future.itertuples(index=False)):
        exit_price = None
        reason = None

        if direction == "long":
            if candle.open <= stop_price:
                exit_price, reason = candle.open, "stop_loss"
            elif candle.open >= target_price:
                exit_price, reason = candle.open, "take_profit"
            elif candle.low <= stop_price:
                exit_price, reason = stop_price, "stop_loss"
            elif candle.high >= target_price:
                exit_price, reason = target_price, "take_profit"
        else:
            if candle.open >= stop_price:
                exit_price, reason = candle.open, "stop_loss"
            elif candle.open <= target_price:
                exit_price, reason = candle.open, "take_profit"
            elif candle.high >= stop_price:
                exit_price, reason = stop_price, "stop_loss"
            elif candle.low <= target_price:
                exit_price, reason = target_price, "take_profit"

        if reason is not None:
            break

    if reason is None:
        offset = len(future) - 1
        candle = future.iloc[-1]
        exit_price = float(candle["close"])
        reason = "time_exit"
        exit_timestamp = candle["timestamp"]
    else:
        exit_timestamp = future.iloc[offset]["timestamp"]

    if direction == "long":
        gross_pct = (exit_price / entry_price - 1.0) * 100
    else:
        gross_pct = (1.0 - exit_price / entry_price) * 100

    return {
        "exit_offset": offset,
        "exit_timestamp": exit_timestamp,
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "gross_return_pct": gross_pct,
        "net_return_pct": gross_pct - ROUND_TRIP_COST_PCT,
    }


def create_walk_forward_splits(df):
    start = int(len(df) * INITIAL_TRAIN_FRACTION)
    remaining = len(df) - start
    if remaining < N_SPLITS * (HORIZON + 2):
        raise ValueError("Not enough test data for the requested splits.")

    fold_size = remaining // N_SPLITS
    splits = []
    for fold in range(1, N_SPLITS + 1):
        end = start + fold_size if fold < N_SPLITS else len(df)
        splits.append((fold, start, end))
        start = end
    return splits

def run_fold(df, split, direction, stop_atr, target_atr):
    fold, test_start, test_end = split
    signals = df[f"{direction}_signal"].to_numpy()
    trades = []
    position = test_start

    while position + 1 + HORIZON <= test_end:
        if not signals[position]:
            position += 1
            continue

        entry_position = position + 1
        future = df.iloc[entry_position:entry_position + HORIZON]
        entry_price = float(future.iloc[0]["open"])
        atr = float(df.iloc[position]["atr_14"])

        outcome = simulate_trade(
            future, entry_price, atr, direction, stop_atr, target_atr
        )
        trades.append({
            "fold": fold,
            "direction": direction,
            "stop_atr": stop_atr,
            "target_atr": target_atr,
            "signal_timestamp": df.iloc[position]["timestamp"],
            "entry_timestamp": future.iloc[0]["timestamp"],
            "entry_price": entry_price,
            "atr": atr,
            "exit_timestamp": outcome["exit_timestamp"],
            "exit_price": outcome["exit_price"],
            "exit_reason": outcome["exit_reason"],
            "gross_return_pct": outcome["gross_return_pct"],
            "net_return_pct": outcome["net_return_pct"],
        })

        # No overlapping positions within a single configuration.
        # The exit candle's close may generate the next signal.
        position = entry_position + outcome["exit_offset"]

    return trades


def run_all_backtests(df):
    all_trades = []
    for split in create_walk_forward_splits(df):
        for direction in ("long", "short"):
            for stop_atr, target_atr in EXIT_COMBINATIONS:
                all_trades.extend(
                    run_fold(df, split, direction, stop_atr, target_atr)
                )
    return pd.DataFrame(all_trades)


def calculate_max_drawdown(returns_pct):
    equity = np.r_[1.0, (1.0 + np.asarray(returns_pct) / 100).cumprod()]
    peaks = np.maximum.accumulate(equity)
    return float(((equity / peaks) - 1.0).min() * 100)


def summarize_results(trades):
    records = []
    for (fold, direction, stop_atr, target_atr), group in trades.groupby(
            ["fold", "direction", "stop_atr", "target_atr"], sort=True
    ):
        returns = group["net_return_pct"].to_numpy()
        equity_end = np.prod(1.0 + returns / 100)
        records.append({
            "fold": fold,
            "direction": direction,
            "stop_atr": stop_atr,
            "target_atr": target_atr,
            "total_trades": len(group),
            "wins": int((returns > 0).sum()),
            "win_rate_pct": float((returns > 0).mean() * 100),
            "average_return_pct": float(returns.mean()),
            "median_return_pct": float(np.median(returns)),
            "total_return_pct": float((equity_end - 1.0) * 100),
            "maximum_drawdown_pct": calculate_max_drawdown(returns),
        })
    return pd.DataFrame(records)


def main():
    print("Loading futures data...")
    df = load_data()

    print("Generating 24-hour breakout signals...")
    df = generate_signals(df)

    print("Running five chronological test folds...")
    trades = run_all_backtests(df)
    if trades.empty:
        print("No eligible breakout trades were generated.")
        return

    # Save each independent configuration's trades, and its summary.
    summary = summarize_results(trades)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trades.to_csv(TRADES_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nWALK-FORWARD FUTURES BREAKOUT BACKTEST")
    print("-------------------------------------")
    print(f"Total candles: {len(df):,}")
    print(f"Total simulated trades across configurations: {len(trades):,}")
    print(f"Walk-forward folds: {N_SPLITS}")
    print("\nPERFORMANCE SUMMARY")
    print(summary.round(4).to_string(index=False))
    print(f"\nTRADE RECORDS SAVED TO:\n{TRADES_FILE}")
    print(f"\nSUMMARY SAVED TO:\n{SUMMARY_FILE}")
    print("\nNote: Full-balance compounded return is illustrative, not risk-sized account P&L.")
    print("Funding, liquidation, and account-level risk are not modeled.")


if __name__ == "__main__":
    main()