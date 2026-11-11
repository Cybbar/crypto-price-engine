
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "btc_features.csv"

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Test whether recent returns relate to the next-hour return.
df["next_hour_return"] = df["target_return_1h"]

tests = {
    "Previous 1-hour return": "return_1h",
    "Previous 24-hour return": "return_24h",
    "24-hour trend": "trend_24h",
    "24-hour volatility": "volatility_24h",
    "Volume ratio": "volume_ratio_24h",
    "Hourly range ratio": "range_ratio",
}

print("RELATIONSHIP WITH NEXT-HOUR BTC RETURN")
print("=" * 65)

for label, feature in tests.items():
    data = df[[feature, "next_hour_return"]].dropna()

    correlation = data[feature].corr(data["next_hour_return"])

    # Compare next-hour returns when the feature is above or below
    # its own median. This is descriptive, not a trading strategy.
    median = data[feature].median()
    high = data.loc[
        data[feature] > median, "next_hour_return"
    ]
    low = data.loc[
        data[feature] <= median, "next_hour_return"
    ]

    print(f"\n{label}")
    print(f"  Correlation:             {correlation:.5f}")
    print(f"  Next return, high group: {high.mean():.6f}")
    print(f"  Next return, low group:  {low.mean():.6f}")
    print(f"  High-group observations: {len(high):,}")
    print(f"  Low-group observations:  {len(low):,}")
