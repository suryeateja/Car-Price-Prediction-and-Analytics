
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DIR = os.path.join("data", "raw")
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

datasets = {
    "car_data.csv": {
        "target": "Selling_Price",
        "year": "Year",
        "km": "Kms_Driven"
    },
    "car_details.csv": {
        "target": "selling_price",
        "year": "year",
        "km": "km_driven"
    },
    "car_details_v4.csv": {
        "target": "Price",
        "year": "Year",
        "km": "Kilometer"
    }
}

for filename, cols in datasets.items():
    df = pd.read_csv(os.path.join(RAW_DIR, filename))
    prefix = filename.replace(".csv", "")

    # 1. Price distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df[cols["target"]], bins=40, kde=True)
    plt.title(f"Price Distribution - {filename}")
    plt.xlabel("Selling Price")
    plt.tight_layout()
    plt.savefig(
        os.path.join(GRAPH_DIR, f"{prefix}_price_distribution.png"),
        dpi=150
    )
    plt.close()

    # 2. Price vs year
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x=cols["year"],
        y=cols["target"],
        alpha=0.6
    )
    plt.title(f"Price vs Year - {filename}")
    plt.xlabel("Manufacturing Year")
    plt.ylabel("Selling Price")
    plt.tight_layout()
    plt.savefig(
        os.path.join(GRAPH_DIR, f"{prefix}_price_vs_year.png"),
        dpi=150
    )
    plt.close()

    # 3. Price vs kilometres driven
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x=cols["km"],
        y=cols["target"],
        alpha=0.6
    )
    plt.title(f"Price vs Kilometres Driven - {filename}")
    plt.xlabel("Kilometres Driven")
    plt.ylabel("Selling Price")
    plt.tight_layout()
    plt.savefig(
        os.path.join(GRAPH_DIR, f"{prefix}_price_vs_kilometres.png"),
        dpi=150
    )
    plt.close()

    # 4. Correlation heatmap for numeric features
    numeric = df.select_dtypes(include="number")
    plt.figure(figsize=(9, 6))
    sns.heatmap(
        numeric.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0
    )
    plt.title(f"Numerical Feature Correlations - {filename}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(GRAPH_DIR, f"{prefix}_correlation_heatmap.png"),
        dpi=150
    )
    plt.close()

    print(f"Created 4 graphs for {filename}")

print("\nEDA visualizations saved in the graphs/ folder.")
