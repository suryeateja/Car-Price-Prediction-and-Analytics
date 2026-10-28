
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# File paths
RAW_PATH = "data/raw/car_details_v4.csv"
PROCESSED_PATH = "data/processed/car_details_v4_cleaned.csv"

RESULTS_DIR = "results"
GRAPH_DIR = "graphs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


def prepare_data(path):
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Remove rows without a target price
    df = df.dropna(subset=["price"])

    X = df.drop(columns=["price"])
    y = pd.to_numeric(df["price"], errors="coerce")

    # Remove rows with invalid target prices
    valid = y.notna() & (y >= 0)
    X = X.loc[valid].copy()
    y = y.loc[valid].astype(float)

    return X, y


def build_model(X_train):
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])


def run_experiment(name, path):
    X, y = prepare_data(path)

    # Consistent split settings
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = build_model(X_train)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n--- {name} ---")
    print(f"Dataset rows: {len(X)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"MAE:  Rs. {mae:,.2f}")
    print(f"RMSE: Rs. {rmse:,.2f}")
    print(f"R²:   {r2:.4f}")

    return {
        "Experiment": name,
        "Rows": len(X),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# Experiment 1: Raw data
raw_result = run_experiment(
    "Raw Data",
    RAW_PATH
)

# Experiment 2: Preprocessed data
processed_result = run_experiment(
    "Preprocessed Data",
    PROCESSED_PATH
)

# Compare results
results_df = pd.DataFrame([raw_result, processed_result])

results_df.to_csv(
    os.path.join(RESULTS_DIR, "raw_vs_preprocessed.csv"),
    index=False
)

print("\n===== FINAL COMPARISON =====")
print(results_df.to_string(index=False))

# Plot comparison
metrics = ["MAE", "RMSE", "R2"]

for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Experiment"], results_df[metric])
    plt.title(f"Raw vs Preprocessed Data: {metric}")
    plt.ylabel(metric)
    plt.tight_layout()

    plt.savefig(
        os.path.join(GRAPH_DIR, f"raw_vs_preprocessed_{metric.lower()}.png"),
        dpi=150
    )

    plt.close()

print("\nSaved results/raw_vs_preprocessed.csv")
print("Saved comparison graphs in the graphs folder.")
