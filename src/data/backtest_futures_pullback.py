from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CANDLE_FILE = (
    ROOT / "data" / "futures" / "raw"
    / "btcusdt_perpetual_1h.csv"
)

REGIME_FILE = (
    ROOT / "data" / "futures" / "processed"
    / "btcusdt_futures_regimes.csv"
)

OUTPUT_DIR = ROOT / "data" / "futures" / "processed"

TRADES_FILE = OUTPUT_DIR / "futures_pullback_trades.csv"
SUMMARY_FILE = OUTPUT_DIR / "futures_pullback_summary.csv"


# Experimental assumptions
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
ROUND_TRIP_COST = 2 * (FEE_RATE + SLIPPAGE_RATE)

ADX_MINIMUM = 25
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
INITIAL_TRAIN_FRACTION = 0.50

def load_data():

    candles = pd.read_csv(
        CANDLE_FILE,
        parse_dates=["timestamp"],
    )

    regimes = pd.read_csv(
        REGIME_FILE,
        parse_dates=["timestamp"],
    )

    candles = candles.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    regimes = regimes.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    required_candles = [
        "timestamp", "open", "high", "low", "close"
    ]

    required_regimes = [
        "timestamp",
        "market_regime",
        "volatility_state",
        "atr_14",
        "adx_14",
        "ema_20",
        "ema_50",
    ]

    for column in required_candles:
        if column not in candles.columns:
            raise ValueError(
                f"Missing candle column: {column}"
            )

    for column in required_regimes:
        if column not in regimes.columns:
            raise ValueError(
                f"Missing regime column: {column}"
            )

    if candles["timestamp"].duplicated().any():
        raise ValueError("Duplicate candle timestamps.")

    if regimes["timestamp"].duplicated().any():
        raise ValueError("Duplicate regime timestamps.")

    price_columns = [
        "open", "high", "low", "close"
    ]

    if not np.isfinite(
        candles[price_columns]
    ).all().all():
        raise ValueError("Invalid candle prices.")

    if (candles[price_columns] <= 0).any().any():
        raise ValueError("Non-positive candle prices.")

    df = candles.merge(
        regimes[required_regimes],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No matching candle and regime data.")

    if not np.isfinite(
        df[["atr_14", "adx_14", "ema_20", "ema_50"]]
    ).all().all():
        raise ValueError("Invalid indicator values.")

    if (df["atr_14"] <= 0).any():
        raise ValueError("Invalid ATR values.")

    return df

def generate_signals(df):

    trending = df["market_regime"].eq("trending")
    strong_trend = df["adx_14"] >= ADX_MINIMUM

    bullish_trend = df["ema_20"] > df["ema_50"]
    bearish_trend = df["ema_20"] < df["ema_50"]

    # Previous candle was on the opposite side of EMA 20
    previous_below_ema = (
        df["close"].shift(1) <= df["ema_20"].shift(1)
    )

    previous_above_ema = (
        df["close"].shift(1) >= df["ema_20"].shift(1)
    )

    # Current candle confirms a return in the trend direction
    bullish_recovery = (
        (df["close"] > df["ema_20"])
        & (df["close"] > df["open"])
    )

    bearish_recovery = (
        (df["close"] < df["ema_20"])
        & (df["close"] < df["open"])
    )

    df["long_signal"] = (
        trending
        & strong_trend
        & bullish_trend
        & previous_below_ema
        & bullish_recovery
    )

    df["short_signal"] = (
        trending
        & strong_trend
        & bearish_trend
        & previous_above_ema
        & bearish_recovery
    )

    return df

def simulate_trade(
    future,
    entry,
    atr,
    direction,
    stop_atr,
    target_atr,
):

    if direction == "long":
        stop_price = entry - stop_atr * atr
        target_price = entry + target_atr * atr
    else:
        stop_price = entry + stop_atr * atr
        target_price = entry - target_atr * atr

    exit_price = None
    exit_reason = None
    exit_timestamp = None

    for candle in future.itertuples(index=False):

        if direction == "long":

            if candle.open <= stop_price:
                exit_price = candle.open
                exit_reason = "stop_loss"

            elif candle.open >= target_price:
                exit_price = candle.open
                exit_reason = "take_profit"

            else:
                stop_hit = candle.low <= stop_price
                target_hit = candle.high >= target_price

                if stop_hit:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                elif target_hit:
                    exit_price = target_price
                    exit_reason = "take_profit"

        else:

            if candle.open >= stop_price:
                exit_price = candle.open
                exit_reason = "stop_loss"

            elif candle.open <= target_price:
                exit_price = candle.open
                exit_reason = "take_profit"

            else:
                stop_hit = candle.high >= stop_price
                target_hit = candle.low <= target_price

                if stop_hit:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                elif target_hit:
                    exit_price = target_price
                    exit_reason = "take_profit"

        if exit_price is not None:
            exit_timestamp = candle.timestamp
            break

    if exit_price is None:
        last_candle = future.iloc[-1]
        exit_price = last_candle["close"]
        exit_timestamp = last_candle["timestamp"]
        exit_reason = "time_exit"

    if direction == "long":
        gross_return = (exit_price / entry) - 1
    else:
        gross_return = 1 - (exit_price / entry)

    net_return = gross_return - ROUND_TRIP_COST

    return {
        "exit_timestamp": exit_timestamp,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_return_pct": gross_return * 100,
        "net_return_pct": net_return * 100,
    }

def create_walk_forward_splits(df):

    total_rows = len(df)

    initial_train_size = int(
        total_rows * INITIAL_TRAIN_FRACTION
    )

    remaining_rows = total_rows - initial_train_size

    fold_size = remaining_rows // N_SPLITS

    splits = []

    for fold in range(N_SPLITS):

        test_start = initial_train_size + (
            fold * fold_size
        )

        if fold == N_SPLITS - 1:
            test_end = total_rows
        else:
            test_end = test_start + fold_size

        splits.append({
            "fold": fold + 1,
            "test_start": test_start,
            "test_end": test_end,
        })

    return splits

def run_fold(df, split, direction, stop_atr, target_atr):

    trades = []

    start = split["test_start"]
    end = split["test_end"]

    position = start

    while position < end - 1:

        row = df.iloc[position]

        if direction == "long":
            signal = row["long_signal"]
        else:
            signal = row["short_signal"]

        if not signal:
            position += 1
            continue

        entry_position = position + 1

        # Require the complete 24-hour trade window
        # to remain inside this test fold.
        if entry_position + HORIZON > end:
            break

        future = df.iloc[
            entry_position:
            entry_position + HORIZON
        ].copy()

        if len(future) != HORIZON:
            break

        # Confirm hourly continuity.
        expected = pd.date_range(
            start=df.iloc[entry_position]["timestamp"],
            periods=HORIZON,
            freq="h",
        )

        if not future["timestamp"].reset_index(
            drop=True
        ).equals(pd.Series(expected)):
            position += 1
            continue

        entry = float(
            df.iloc[entry_position]["open"]
        )

        atr = float(row["atr_14"])

        outcome = simulate_trade(
            future=future,
            entry=entry,
            atr=atr,
            direction=direction,
            stop_atr=stop_atr,
            target_atr=target_atr,
        )

        exit_position = df.index[
            df["timestamp"] == outcome["exit_timestamp"]
        ][0]

        trades.append({
            "fold": split["fold"],
            "direction": direction,
            "stop_atr": stop_atr,
            "target_atr": target_atr,
            "signal_timestamp": row["timestamp"],
            "entry_timestamp": df.iloc[
                entry_position
            ]["timestamp"],
            "entry_price": entry,
            "atr": atr,
            **outcome,
        })

        # The next signal can only be used after
        # the exit candle has completed.
        position = exit_position

    return trades

def run_all_backtests(df):

    all_trades = []

    splits = create_walk_forward_splits(df)

    for split in splits:

        for direction in ["long", "short"]:

            for stop_atr, target_atr in EXIT_COMBINATIONS:

                trades = run_fold(
                    df=df,
                    split=split,
                    direction=direction,
                    stop_atr=stop_atr,
                    target_atr=target_atr,
                )

                all_trades.extend(trades)

    if not all_trades:
        raise ValueError(
            "No trades generated. Check entry signals and data."
        )

    return pd.DataFrame(all_trades)

def calculate_max_drawdown(returns):

    equity = (1 + returns / 100).cumprod()

    running_peak = equity.cummax()

    drawdown = (
        equity / running_peak - 1
    ) * 100

    return drawdown.min()


def summarize_results(trades):

    summary = (
        trades.groupby(
            [
                "fold",
                "direction",
                "stop_atr",
                "target_atr",
            ],
            observed=True,
        )
        .agg(
            total_trades=("net_return_pct", "count"),
            wins=(
                "net_return_pct",
                lambda x: (x > 0).sum(),
            ),
            win_rate_pct=(
                "net_return_pct",
                lambda x: (x > 0).mean() * 100,
            ),
            average_return_pct=(
                "net_return_pct",
                "mean",
            ),
            median_return_pct=(
                "net_return_pct",
                "median",
            ),
            total_return_pct=(
                "net_return_pct",
                lambda x: (
                    (1 + x / 100).prod() - 1
                ) * 100,
            ),
            maximum_drawdown_pct=(
                "net_return_pct",
                calculate_max_drawdown,
            ),
        )
        .reset_index()
    )

    return summary.round(4)

def main():

    print("Loading futures data...")

    df = load_data()

    print("Generating entry signals...")

    df = generate_signals(df)

    print("Creating walk-forward tests...")

    trades = run_all_backtests(df)

    print("Calculating performance summaries...")

    summary = summarize_results(trades)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        TRADES_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("\nWALK-FORWARD FUTURES BACKTEST")
    print("--------------------------------")

    print("Total candles:", len(df))
    print("Total trades:", len(trades))
    print("Walk-forward folds:", N_SPLITS)

    print("\nPERFORMANCE SUMMARY")

    print(
        summary.to_string(index=False)
    )

    print("\nTRADE RECORDS SAVED TO:")
    print(TRADES_FILE)

    print("\nSUMMARY SAVED TO:")
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()