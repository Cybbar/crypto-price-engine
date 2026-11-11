from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "futures" / "processed"
    / "btcusdt_futures_regimes.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "futures" / "processed"
    / "futures_regime_evaluation.csv"
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_regimes():

    print("Loading regime data...")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "market_regime",
        "volatility_state",
        "target_return_1h",
        "target_return_24h",
    ]

    if df[required].isna().any().any():
        raise ValueError("Missing values in evaluation columns.")

    if not np.isfinite(
        df[["target_return_1h", "target_return_24h"]]
        .to_numpy()
    ).all():
        raise ValueError("Infinite target values detected.")

    # --------------------------------------------------------
    # GROUP BY REGIME AND VOLATILITY
    # --------------------------------------------------------

    results = (
        df.groupby(
            ["market_regime", "volatility_state"],
            observed=True,
        )
        .agg(
            observations=("target_return_1h", "size"),

            mean_return_1h=("target_return_1h", "mean"),
            median_return_1h=("target_return_1h", "median"),
            positive_1h=(
                "target_return_1h",
                lambda x: (x > 0).mean() * 100,
            ),

            mean_return_24h=("target_return_24h", "mean"),
            median_return_24h=("target_return_24h", "median"),
            positive_24h=(
                "target_return_24h",
                lambda x: (x > 0).mean() * 100,
            ),
        )
        .reset_index()
    )

    # Express returns as percentages.
    return_columns = [
        "mean_return_1h",
        "median_return_1h",
        "mean_return_24h",
        "median_return_24h",
    ]

    results[return_columns] *= 100

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("\nREGIME PERFORMANCE")
    print(results.round(4).to_string(index=False))

    print("\nTOTAL OBSERVATIONS:", len(df))

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    evaluate_regimes()