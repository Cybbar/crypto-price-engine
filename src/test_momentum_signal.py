
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "btc_usdt_1h.csv"
OUTPUT = ROOT / "reports" / "momentum_signal_diagnostics.csv"


def main():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.reset_index(drop=True)

    close = df["close"]

    # Information available at the end of the current candle.
    df["past_24h_return"] = np.log(close / close.shift(24))

    # Returns occurring AFTER the current candle closes.
    for horizon in [4, 12, 24]:
        df[f"future_{horizon}h_return"] = np.log(
            close.shift(-horizon) / close
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    results = []

    for horizon in [4, 12, 24]:
        future_col = f"future_{horizon}h_return"
        sample = df[["past_24h_return", future_col]].dropna()

        # Correlation is descriptive, not proof of predictive power.
        correlation = sample["past_24h_return"].corr(
            sample[future_col]
        )

        # Define groups using past returns only.
        low_cut = sample["past_24h_return"].quantile(0.25)
        high_cut = sample["past_24h_return"].quantile(0.75)

        low = sample[sample["past_24h_return"] <= low_cut][future_col]
        high = sample[sample["past_24h_return"] >= high_cut][future_col]

        print("\n" + "=" * 65)
        print(f"FUTURE {horizon}-HOUR RETURN")
        print(f"Observations: {len(sample):,}")
        print(f"Correlation with past 24h return: {correlation:.5f}")
        print(f"Bottom-quartile mean future return: {low.mean():.4%}")
        print(f"Top-quartile mean future return:    {high.mean():.4%}")
        print(f"Top minus bottom:                   {high.mean() - low.mean():.4%}")

        results.append({
            "horizon_hours": horizon,
            "observations": len(sample),
            "correlation": correlation,
            "bottom_quartile_mean": low.mean(),
            "top_quartile_mean": high.mean(),
            "top_minus_bottom": high.mean() - low.mean(),
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT, index=False)
    print(f"\nSaved results to: {OUTPUT}")
    print(
        "\nNote: this is an exploratory full-sample analysis. "
        "It is not an out-of-sample trading test."
    )


if __name__ == "__main__":
    main()
