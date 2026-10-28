import pandas as pd
import numpy as np

raw_path = "data/raw/car_details_v4.csv"
processed_path = "data/processed/car_details_v4_cleaned.csv"

raw = pd.read_csv(raw_path)
processed = pd.read_csv(processed_path)

# Standardize column names
raw.columns = raw.columns.str.strip().str.lower()
processed.columns = processed.columns.str.strip().str.lower()

print("RAW SHAPE:", raw.shape)
print("PROCESSED SHAPE:", processed.shape)

print("\nSame column names?")
print(set(raw.columns) == set(processed.columns))

print("\nMissing cells in raw data:", raw.isna().sum().sum())
print("Missing cells in processed data:", processed.isna().sum().sum())

# Compare corresponding values after aligning columns
common_columns = sorted(set(raw.columns) & set(processed.columns))

raw_aligned = raw[common_columns].reset_index(drop=True)
processed_aligned = processed[common_columns].reset_index(drop=True)

print("\nAre the datasets identical?")
print(raw_aligned.equals(processed_aligned))

# Count differing cells
if raw_aligned.shape == processed_aligned.shape:
    differences = raw_aligned.ne(processed_aligned)

    # Treat NaN vs NaN as equal
    differences = differences & ~(
        raw_aligned.isna() & processed_aligned.isna()
    )

    print("Number of differing cells:", differences.sum().sum())

print("\nRaw missing values by column:")
print(raw.isna().sum()[raw.isna().sum() > 0])

print("\nProcessed missing values by column:")
print(processed.isna().sum()[processed.isna().sum() > 0])
