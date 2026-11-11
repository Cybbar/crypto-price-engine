from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "futures" / "processed"
    / "btcusdt_futures_regimes.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "futures" / "processed"
    / "futures_regime_stability.csv"
)

DATE_COLUMN = "timestamp"

REGIME_COLUMNS = [
    "market_regime",
    "volatility_state",
]

TARGET_COLUMN = "target_return_24h"


def evaluate_stability():
    df = pd.read_csv(INPUT_FILE, parse_dates=[DATE_COLUMN])

    required = [
        DATE_COLUMN,
        *REGIME_COLUMNS,
        TARGET_COLUMN,
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values(DATE_COLUMN).copy()

    if df[required].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    if not np.isfinite(df[TARGET_COLUMN]).all():
        raise ValueError("Infinite target values found.")

    # Sample once per UTC day to reduce overlap between 24h targets.
    df["date"] = df[DATE_COLUMN].dt.floor("D")
    daily = df.drop_duplicates(subset="date", keep="first").copy()

    # Divide the history into six-month chronological periods.
    daily["period"] = (
            daily[DATE_COLUMN].dt.year.astype(str)
            + "-H"
            + np.where(daily[DATE_COLUMN].dt.month <= 6, "1", "2")
    )
    results = (
        daily.groupby(["period", *REGIME_COLUMNS], observed=True)
        .agg(
            observations=(TARGET_COLUMN, "count"),
            mean_return_24h=(TARGET_COLUMN, "mean"),
            median_return_24h=(TARGET_COLUMN, "median"),
            positive_24h=(
                TARGET_COLUMN,
                lambda x: (x > 0).mean() * 100,
            ),
        )
        .reset_index()
    )

    results["mean_return_24h"] *= 100
    results["median_return_24h"] *= 100

    results = results.round(4)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nREGIME STABILITY RESULTS")
    print(results.to_string(index=False))

    print("\nTOTAL DAILY OBSERVATIONS:", len(daily))
    print("PERIODS:", daily["period"].nunique())
    print("SAVED TO:", OUTPUT_FILE)


if __name__ == "__main__":
    evaluate_stability()
