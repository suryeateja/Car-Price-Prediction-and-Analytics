
import os
import pandas as pd

# Project paths
RAW_DATA_DIR = os.path.join("data", "raw")
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset files
dataset_files = [
    "car_data.csv",
    "car_details.csv",
    "car_details_v4.csv"
]

report_path = os.path.join(RESULTS_DIR, "dataset_inspection.txt")

with open(report_path, "w", encoding="utf-8") as report:

    for filename in dataset_files:
        filepath = os.path.join(RAW_DATA_DIR, filename)

        print("\n" + "=" * 70)
        print(f"DATASET: {filename}")
        print("=" * 70)

        report.write("\n" + "=" * 70 + "\n")
        report.write(f"DATASET: {filename}\n")
        report.write("=" * 70 + "\n")

        if not os.path.exists(filepath):
            message = f"FILE NOT FOUND: {filepath}"
            print(message)
            report.write(message + "\n")
            continue

        try:
            df = pd.read_csv(filepath)

            # 1. Dataset dimensions
            shape_info = (
                f"\nRows: {df.shape[0]}\n"
                f"Columns: {df.shape[1]}\n"
            )
            print(shape_info)
            report.write(shape_info)

            # 2. Column names and data types
            print("\nCOLUMN NAMES AND DATA TYPES:")
            print(df.dtypes)

            report.write("\nCOLUMN NAMES AND DATA TYPES:\n")
            report.write(df.dtypes.to_string() + "\n")

            # 3. First five rows
            print("\nFIRST FIVE ROWS:")
            print(df.head())

            report.write("\nFIRST FIVE ROWS:\n")
            report.write(df.head().to_string() + "\n")

            # 4. Missing values
            missing = df.isnull().sum()
            missing = missing[missing > 0]

            print("\nMISSING VALUES:")
            if missing.empty:
                print("No missing values found.")
                report.write("\nMISSING VALUES: None\n")
            else:
                print(missing.sort_values(ascending=False))
                report.write(
                    "\nMISSING VALUES:\n"
                    + missing.sort_values(
                        ascending=False
                    ).to_string()
                    + "\n"
                )

            # 5. Duplicate rows
            duplicate_count = int(df.duplicated().sum())
            print(f"\nExact duplicate rows: {duplicate_count}")

            report.write(
                f"\nExact duplicate rows: {duplicate_count}\n"
            )

            # 6. Numeric statistics
            print("\nNUMERIC COLUMN STATISTICS:")
            numeric_stats = df.describe().T
            print(numeric_stats)

            report.write("\nNUMERIC COLUMN STATISTICS:\n")
            report.write(numeric_stats.to_string() + "\n")

            # 7. Unique values in categorical columns
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns

            print("\nCATEGORICAL COLUMN SUMMARY:")

            for col in categorical_cols:
                unique_count = df[col].nunique(dropna=True)

                print(
                    f"{col}: {unique_count} unique values"
                )

                report.write(
                    f"{col}: {unique_count} unique values\n"
                )

            print(f"\nInspection completed for {filename}")

        except Exception as error:
            message = f"ERROR reading {filename}: {error}"
            print(message)
            report.write(message + "\n")

print("\n" + "=" * 70)
print("ALL DATASETS INSPECTED")
print(f"Report saved to: {report_path}")
print("=" * 70)
