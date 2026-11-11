
import pandas as pd

df = pd.read_csv("reports/regime_offset_results.csv")

pivot = df.pivot_table(
    index=["fold", "offset"],
    columns="regime",
    values="avg_future_volatility_pct"
)

counts = df.pivot_table(
    index=["fold", "offset"],
    columns="regime",
    values="observations"
)

print("\nSAMPLE SIZE AND VOLATILITY DIFFERENCE")
print("=" * 75)

for fold in range(1, 6):
    print(f"\nFOLD {fold}")

    fold_values = pivot.loc[fold]
    fold_counts = counts.loc[fold]

    differences = (
        fold_values["high_volatility"]
        - fold_values[["trending", "ranging"]].max(axis=1)
    )

    print(
        "High-volatility observations per offset:",
        f"min={fold_counts['high_volatility'].min():.0f},",
        f"median={fold_counts['high_volatility'].median():.0f},",
        f"max={fold_counts['high_volatility'].max():.0f}"
    )

    print(
        "Median volatility advantage:",
        f"{differences.median():.4f} percentage points"
    )

    print(
        "Offsets with at least 5 high-volatility observations:",
        f"{(fold_counts['high_volatility'] >= 5).sum()}/12"
    )
