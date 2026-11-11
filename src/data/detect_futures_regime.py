from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "futures" / "processed"
    / "btcusdt_futures_features.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "futures" / "processed"
    / "btcusdt_futures_regimes.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

ADX_TRENDING = 25
ADX_RANGING = 20

HIGH_VOLATILITY_THRESHOLD = 1.5


# ============================================================
# DETECT MARKET REGIMES
# ============================================================

def detect_regimes():

    print("Loading futures features...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
    )

    required_columns = [
        "adx_14",
        "volatility_ratio",
        "ema_trend",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values found in regime inputs."
        )

    if not np.isfinite(
        df[required_columns].to_numpy()
    ).all():
        raise ValueError(
            "Infinite values found in regime inputs."
        )

    # --------------------------------------------------------
    # BASE MARKET REGIME
    # --------------------------------------------------------

    df["market_regime"] = np.select(
        [
            df["adx_14"] >= ADX_TRENDING,
            df["adx_14"] < ADX_RANGING,
        ],
        [
            "trending",
            "ranging",
        ],
        default="transitioning",
    )

    # --------------------------------------------------------
    # VOLATILITY CONDITION
    # --------------------------------------------------------

    df["volatility_state"] = np.where(
        df["volatility_ratio"]
        >= HIGH_VOLATILITY_THRESHOLD,
        "high",
        "normal",
    )

    # --------------------------------------------------------
    # TREND DIRECTION
    # --------------------------------------------------------

    df["trend_direction"] = np.select(
        [
            (
                (df["market_regime"] == "trending")
                & (df["ema_trend"] > 0)
            ),
            (
                (df["market_regime"] == "trending")
                & (df["ema_trend"] < 0)
            ),
        ],
        [
            "bullish",
            "bearish",
        ],
        default="neutral",
    )

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

    print("\nREGIME DETECTION COMPLETE")

    print("\nMarket regime distribution:")
    print(
        df["market_regime"]
        .value_counts()
    )

    print("\nVolatility distribution:")
    print(
        df["volatility_state"]
        .value_counts()
    )

    print("\nTrend direction distribution:")
    print(
        df["trend_direction"]
        .value_counts()
    )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    detect_regimes()