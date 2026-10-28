
import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

input_path = os.path.join(RAW_DIR, "car_details_v4.csv")
output_path = os.path.join(
    PROCESSED_DIR, "car_details_v4_cleaned.csv"
)
outlier_path = os.path.join(RESULTS_DIR, "outlier_analysis.csv")
report_path = os.path.join(RESULTS_DIR, "preprocessing_report.txt")

# Load raw dataset
df = pd.read_csv(input_path)

original_shape = df.shape
report = [
    "CAR PRICE PREDICTION - PREPROCESSING REPORT",
    "=" * 55,
    f"Original shape: {original_shape}"
]

# 1. Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

# 2. Remove exact duplicates
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

report.append(f"Exact duplicates removed: {duplicates_removed}")

# 3. Convert important fields to numeric
for col in ["price", "year", "kilometer"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Remove invalid target prices
invalid_price = int(
    (df["price"].isna() | (df["price"] <= 0)).sum()
)

df = df[df["price"].notna() & (df["price"] > 0)].copy()

report.append(f"Invalid/missing target rows removed: {invalid_price}")

# 5. Validate vehicle year
invalid_year = int(
    (~df["year"].between(1980, 2026)).sum()
)

df = df[df["year"].between(1980, 2026)].copy()

report.append(f"Invalid year rows removed: {invalid_year}")

# 6. Remove negative kilometres
negative_km = int((df["kilometer"] < 0).sum())

df = df[
    df["kilometer"].isna() | (df["kilometer"] >= 0)
].copy()

report.append(f"Negative kilometre rows removed: {negative_km}")

# 7. Convert blank text values into missing values
object_cols = df.select_dtypes(include="object").columns

for col in object_cols:
    df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)

missing_before = int(df.isna().sum().sum())
report.append(f"Missing cells before imputation: {missing_before}")

# 8. Analyze numerical outliers before imputation
outlier_rows = []

numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    values = df[col].dropna()

    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = int(((df[col] < lower) | (df[col] > upper)).sum())

    outlier_rows.append({
        "Feature": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower,
        "Upper_Bound": upper,
        "Outlier_Count": count
    })

pd.DataFrame(outlier_rows).to_csv(outlier_path, index=False)

report.append(
    "Potential numerical outliers identified and retained."
)

# 9. Fill missing numerical feature values with their median
numeric_features = [
    col for col in df.select_dtypes(include="number").columns
    if col != "price"
]

for col in numeric_features:
    median_value = df[col].median()

    if pd.notna(median_value):
        df[col] = df[col].fillna(median_value)

# 10. Fill missing categorical feature values with Unknown
categorical_features = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for col in categorical_features:
    df[col] = df[col].fillna("Unknown")

# 11. Save processed dataset
df.to_csv(output_path, index=False)

missing_after = int(df.isna().sum().sum())

report.append(f"Missing cells after imputation: {missing_after}")
report.append(f"Final shape: {df.shape}")
report.append(f"Original shape: {original_shape}")
report.append(f"Processed shape: {df.shape}")
report.append("Numerical feature missing values: median imputation.")
report.append("Categorical feature missing values: Unknown.")
report.append("Target price was not imputed.")
report.append(
    "Potential outliers were retained because unusual values "
    "may represent legitimate vehicles."
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report))

print("\n".join(report))

print("\nSaved processed dataset:", output_path)
print("Saved outlier analysis:", outlier_path)
print("Saved report:", report_path)
