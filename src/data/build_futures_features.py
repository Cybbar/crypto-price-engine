
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CANDLE_FILE = (
    ROOT / "data" / "futures" / "raw"
    / "btcusdt_perpetual_1h.csv"
)

FUNDING_FILE = (
    ROOT / "data" / "futures" / "raw"
    / "btcusdt_funding.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "futures" / "processed"
    / "btcusdt_futures_features.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

ATR_PERIOD = 14
ADX_PERIOD = 14

EMA_FAST = 20
EMA_SLOW = 50

VOLATILITY_WINDOW = 24
FUNDING_LOOKBACK = 3


# ============================================================
# INDICATORS
# ============================================================

def calculate_atr(df, period=14):

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def calculate_adx(df, period=14):

    high_difference = df["high"].diff()
    low_difference = -df["low"].diff()

    plus_dm = pd.Series(
        np.where(
            (high_difference > low_difference)
            & (high_difference > 0),
            high_difference,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (low_difference > high_difference)
            & (low_difference > 0),
            low_difference,
            0.0,
        ),
        index=df.index,
    )

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        / atr
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        / atr
    )

    denominator = (plus_di + minus_di).replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / denominator

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features():

    print("Loading futures candles...")

    df = pd.read_csv(CANDLE_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # PRICE RETURNS
    # --------------------------------------------------------

    df["return_1h"] = np.log(
        df["close"] / df["close"].shift(1)
    )

    df["return_24h"] = np.log(
        df["close"] / df["close"].shift(24)
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    df["volatility_24h"] = (
        df["return_1h"].rolling(24).std()
    )

    df["volatility_7d"] = (
        df["return_1h"].rolling(168).std()
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    previous_volume_avg = (
        df["volume"].rolling(24).mean().shift(1)
    )

    df["volume_ratio_24h"] = (
        df["volume"] / previous_volume_avg
    )

    # --------------------------------------------------------
    # PRICE RANGE
    # --------------------------------------------------------

    df["range_ratio"] = (
        (df["high"] - df["low"]) / df["close"]
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    previous_price_avg = (
        df["close"].rolling(24).mean().shift(1)
    )

    df["trend_24h"] = (
        df["close"] / previous_price_avg - 1
    )

    df["ema_20"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False,
    ).mean()

    df["ema_50"] = df["close"].ewm(
        span=EMA_SLOW,
        adjust=False,
    ).mean()

    df["ema_trend"] = (
        df["ema_20"] / df["ema_50"] - 1
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr_14"] = calculate_atr(
        df,
        ATR_PERIOD,
    )

    df["atr_percentage"] = (
        df["atr_14"] / df["close"]
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    df["adx_14"] = calculate_adx(
        df,
        ADX_PERIOD,
    )

    # --------------------------------------------------------
    # VOLATILITY REGIME
    # --------------------------------------------------------

    volatility_baseline = (
        df["volatility_24h"]
        .rolling(168, min_periods=168)
        .median()
    )

    df["volatility_ratio"] = (
        df["volatility_24h"] / volatility_baseline
    )

    # --------------------------------------------------------
    # FUNDING
    # --------------------------------------------------------

    funding = pd.read_csv(FUNDING_FILE)

    funding["timestamp"] = pd.to_datetime(
        funding["timestamp"],
        utc=True,
    )

    funding = (
        funding.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # Calculate the average of the last 3 actual
    # funding events (not the last 3 hourly candles).
    funding["funding_mean_3"] = (
        funding["funding_rate"]
        .rolling(FUNDING_LOOKBACK)
        .mean()
    )

    # Merge only funding information already available
    # at each candle's timestamp.
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        funding[
            ["timestamp", "funding_rate", "funding_mean_3"]
        ],
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    # --------------------------------------------------------
    # FUTURE TARGETS
    # --------------------------------------------------------

    df["target_return_1h"] = np.log(
        df["close"].shift(-1) / df["close"]
    )

    df["target_return_24h"] = np.log(
        df["close"].shift(-24) / df["close"]
    )

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    feature_columns = [
        "return_1h",
        "return_24h",
        "volatility_24h",
        "volatility_7d",
        "volume_ratio_24h",
        "range_ratio",
        "trend_24h",
        "ema_trend",
        "atr_14",
        "atr_percentage",
        "adx_14",
        "volatility_ratio",
        "funding_rate",
        "funding_mean_3",
    ]

    target_columns = [
        "target_return_1h",
        "target_return_24h",
    ]

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    df = df.dropna(
        subset=feature_columns + target_columns
    ).copy()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nFEATURE ENGINEERING COMPLETE")
    print("Rows:", len(df))
    print("First:", df["timestamp"].min())
    print("Last:", df["timestamp"].max())
    print("Saved:", OUTPUT_FILE)

    print("\nFeature preview:")

    print(
        df[
            ["timestamp"] + feature_columns
            ].tail().to_string(index=False)
    )

    print("\nMissing feature values:")
    print(df[feature_columns].isna().sum())


if __name__ == "__main__":
    build_features()