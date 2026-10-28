
import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

files = {
    "car_data.csv": "Selling_Price",
    "car_details.csv": "selling_price",
    "car_details_v4.csv": "Price"
}

report_path = os.path.join(
    RESULTS_DIR, "data_quality_analysis.txt"
)

with open(report_path, "w", encoding="utf-8") as report:

    for filename, target in files.items():
        path = os.path.join(RAW_DIR, filename)
        df = pd.read_csv(path)

        print("\n" + "=" * 65)
        print(f"DATA QUALITY ANALYSIS: {filename}")
        print("=" * 65)

        report.write(f"\n{'=' * 65}\n{filename}\n")

        # 1. Missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        print("\n1. Missing values:")
        if missing.empty:
            print("No missing values.")
            report.write("\nMissing values: None\n")
        else:
            print(missing.sort_values(ascending=False))
            report.write(
                "\nMissing values:\n"
                + missing.sort_values(
                    ascending=False
                ).to_string()
                + "\n"
            )

        # 2. Duplicate analysis
        duplicates = int(df.duplicated().sum())
        duplicate_percentage = duplicates / len(df) * 100

        print("\n2. Duplicate rows:", duplicates)
        print(
            f"Duplicate percentage: "
            f"{duplicate_percentage:.2f}%"
        )

        report.write(
            f"\nDuplicate rows: {duplicates}\n"
            f"Duplicate percentage: "
            f"{duplicate_percentage:.2f}%\n"
        )

        # 3. Target variable analysis
        print(f"\n3. Target variable: {target}")
        print(df[target].describe())

        report.write(
            f"\nTarget variable: {target}\n"
            + df[target].describe().to_string()
            + "\n"
        )

        # 4. Potential invalid numerical values
        numeric_cols = df.select_dtypes(
            include="number"
        ).columns

        print("\n4. Numerical columns with negative values:")
        negative_found = False

        for col in numeric_cols:
            count = int((df[col] < 0).sum())

            if count > 0:
                negative_found = True
                print(f"{col}: {count}")
                report.write(
                    f"\nNegative values in {col}: {count}\n"
                )

        if not negative_found:
            print("No negative numerical values found.")
            report.write(
                "\nNo negative numerical values found.\n"
            )

        # 5. Potential extreme values using IQR
        print("\n5. Potential outliers (IQR method):")

        for col in numeric_cols:
            series = df[col].dropna()

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            count = int(
                ((series < lower) | (series > upper)).sum()
            )

            if count > 0:
                line = (
                    f"{col}: {count} potential outliers "
                    f"(bounds: {lower:.2f} to {upper:.2f})"
                )
                print(line)
                report.write(line + "\n")

        # 6. Suspicious zero values
        print("\n6. Zero values in numerical columns:")

        for col in numeric_cols:
            count = int((df[col] == 0).sum())

            if count > 0:
                print(f"{col}: {count}")
                report.write(
                    f"Zero values in {col}: {count}\n"
                )

print("\nAnalysis complete.")
print(f"Report saved to: {report_path}")
