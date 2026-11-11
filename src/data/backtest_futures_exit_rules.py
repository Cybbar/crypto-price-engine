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

OUTPUT_FILE = (
    ROOT / "data" / "futures" / "processed"
    / "futures_exit_rule_backtest.csv"
)

# Experimental assumptions, not confirmed exchange costs.
HORIZON = 24
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005

EXIT_COMBINATIONS = [
    (1.0, 1.5),
    (1.0, 2.0),
    (1.5, 2.0),
    (1.5, 3.0),
    (2.0, 3.0),
    (2.0, 4.0),
]

ROUND_TRIP_COST = 2 * (FEE_RATE + SLIPPAGE_RATE)


def simulate_trade(future, entry, atr, direction, stop_atr, target_atr):

    if direction == "long":
        stop_price = entry - stop_atr * atr
        target_price = entry + target_atr * atr
    else:
        stop_price = entry + stop_atr * atr
        target_price = entry - target_atr * atr

    exit_price = None
    exit_reason = None

    for candle in future.itertuples(index=False):

        # Handle gaps at the candle open first.
        if direction == "long":

            if candle.open <= stop_price:
                exit_price = candle.open
                exit_reason = "stop_loss"
                break

            if candle.open >= target_price:
                exit_price = candle.open
                exit_reason = "take_profit"
                break

            stop_hit = candle.low <= stop_price
            target_hit = candle.high >= target_price

        else:

            if candle.open >= stop_price:
                exit_price = candle.open
                exit_reason = "stop_loss"
                break

            if candle.open <= target_price:
                exit_price = candle.open
                exit_reason = "take_profit"
                break

            stop_hit = candle.high >= stop_price
            target_hit = candle.low <= target_price

        # Conservative assumption when both levels are touched.
        if stop_hit:
            exit_price = stop_price
            exit_reason = "stop_loss"
            break

        if target_hit:
            exit_price = target_price
            exit_reason = "take_profit"
            break

    # Exit at the final candle close if neither level was reached.
    if exit_price is None:
        exit_price = future.iloc[-1]["close"]
        exit_reason = "time_exit"

    if direction == "long":
        gross_return = (exit_price / entry) - 1
    else:
        gross_return = 1 - (exit_price / entry)

    # Approximate round-trip fees and slippage.
    net_return = gross_return - ROUND_TRIP_COST

    return {
        "exit_reason": exit_reason,
        "gross_return_pct": gross_return * 100,
        "net_return_pct": net_return * 100,
    }


def backtest():

    candles = pd.read_csv(
        CANDLE_FILE,
        parse_dates=["timestamp"],
    )

    regimes = pd.read_csv(
        REGIME_FILE,
        parse_dates=["timestamp"],
    )

    candles = candles.sort_values("timestamp").reset_index(drop=True)
    regimes = regimes.sort_values("timestamp").reset_index(drop=True)

    required_candles = [
        "timestamp", "open", "high", "low", "close"
    ]

    required_regimes = [
        "timestamp",
        "market_regime",
        "volatility_state",
        "atr_14",
    ]

    for column in required_candles:
        if column not in candles.columns:
            raise ValueError(f"Missing candle column: {column}")

    for column in required_regimes:
        if column not in regimes.columns:
            raise ValueError(f"Missing regime column: {column}")

    if candles["timestamp"].duplicated().any():
        raise ValueError("Duplicate candle timestamps found.")

    if regimes["timestamp"].duplicated().any():
        raise ValueError("Duplicate regime timestamps found.")

    price_columns = ["open", "high", "low", "close"]

    if not np.isfinite(candles[price_columns]).all().all():
        raise ValueError("Invalid candle prices found.")

    if (
        (candles[price_columns] <= 0).any().any()
        or (regimes["atr_14"] <= 0).any()
    ):
        raise ValueError("Invalid prices or ATR values found.")

    # Attach regime and entry ATR to completed candles.
    df = candles.merge(
        regimes[required_regimes],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    # One entry per UTC day.
    df["date"] = df["timestamp"].dt.floor("D")
    entries = df.drop_duplicates(
        subset="date",
        keep="first",
    ).copy()

    # Use only entries with 24 subsequent hourly candles.
    results = []

    for position, (_, row) in enumerate(entries.iterrows()):

        entry_time = row["timestamp"]

        # Locate the entry candle in the full hourly dataset.
        candle_position = candles.index[
            candles["timestamp"] == entry_time
            ][0]

        future = candles.iloc[
            candle_position + 1:
            candle_position + 1 + HORIZON
        ]

        if len(future) != HORIZON:
            continue

        # Reject incomplete or irregular hourly windows.
        expected_timestamps = pd.date_range(
            start=entry_time + pd.Timedelta(hours=1),
            periods=HORIZON,
            freq="h",
        )

        if not future["timestamp"].reset_index(drop=True).equals(
            pd.Series(expected_timestamps)
        ):
            continue

        entry = float(row["close"])
        atr = float(row["atr_14"])

        period = (
            f"{entry_time.year}-H"
            f"{1 if entry_time.month <= 6 else 2}"
        )

        for direction in ["long", "short"]:

            for stop_atr, target_atr in EXIT_COMBINATIONS:

                outcome = simulate_trade(
                    future=future,
                    entry=entry,
                    atr=atr,
                    direction=direction,
                    stop_atr=stop_atr,
                    target_atr=target_atr,
                )

                results.append({
                    "timestamp": entry_time,
                    "period": period,
                    "market_regime": row["market_regime"],
                    "volatility_state": row["volatility_state"],
                    "direction": direction,
                    "stop_atr": stop_atr,
                    "target_atr": target_atr,
                    **outcome,
                })

    trades = pd.DataFrame(results)

    if trades.empty:
        raise ValueError("No trades were generated.")

    # Summarise by period, regime, direction and exit configuration.
    summary = (
        trades.groupby(
            [
                "period",
                "market_regime",
                "volatility_state",
                "direction",
                "stop_atr",
                "target_atr",
            ],
            observed=True,
        )
        .agg(
            observations=("net_return_pct", "count"),
            stop_losses=(
                "exit_reason",
                lambda x: (x == "stop_loss").sum(),
            ),
            take_profits=(
                "exit_reason",
                lambda x: (x == "take_profit").sum(),
            ),
            time_exits=(
                "exit_reason",
                lambda x: (x == "time_exit").sum(),
            ),
            win_rate=(
                "net_return_pct",
                lambda x: (x > 0).mean() * 100,
            ),
            average_net_return=("net_return_pct", "mean"),
            median_net_return=("net_return_pct", "median"),
        )
        .reset_index()
        .round(4)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nFUTURES EXIT RULE BACKTEST")
    print("Round-trip cost assumption:", ROUND_TRIP_COST * 100, "%")
    print("Horizon:", HORIZON, "hours")
    print("Daily entries:", entries.shape[0])

    print("\nRESULTS")
    print(summary.to_string(index=False))

    print("\nTOTAL SIMULATED TRADES:", len(trades))
    print("SAVED TO:", OUTPUT_FILE)


if __name__ == "__main__":
    backtest()
