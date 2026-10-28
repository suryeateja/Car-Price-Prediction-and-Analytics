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

RAW_PATH = "data/raw/car_details_v4.csv"
PROCESSED_PATH = "data/processed/car_details_v4_cleaned.csv"

RESULTS_DIR = "results"
GRAPH_DIR = "graphs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# --------------------------------------------------
# 1. Load and standardize both datasets
# --------------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0].copy()

    return df.reset_index(drop=True)


raw_df = load_data(RAW_PATH)
processed_df = load_data(PROCESSED_PATH)

# Check row alignment before comparing the datasets.
if len(raw_df) != len(processed_df):
    raise ValueError(
        "Raw and processed datasets have different row counts. "
        "Check their row correspondence before comparing models."
    )

if list(raw_df.columns) != list(processed_df.columns):
    raise ValueError(
        "Raw and processed datasets have different columns."
    )

# Both files originate from the same dataset and preserve row order.
# Use the same row indices for both experiments.
indices = np.arange(len(raw_df))

train_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=42
)

# --------------------------------------------------
# 2. Build the same model pipeline for each experiment
# --------------------------------------------------

def make_pipeline(X_train):
    numeric_cols = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

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


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n--- {name} ---")
    print(f"MAE:  Rs. {mae:,.2f}")
    print(f"RMSE: Rs. {rmse:,.2f}")
    print(f"R2:   {r2:.4f}")

    return {
        "Experiment": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


results = []

# --------------------------------------------------
# 3. Required Phase 2 comparison:
#    Raw data vs preprocessed data
# --------------------------------------------------

for name, df in [
    ("Raw Data", raw_df),
    ("Preprocessed Data", processed_df)
]:
    X = df.drop(columns=["price"])
    y = df["price"].astype(float)

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    model = make_pipeline(X_train)

    results.append(evaluate(
        name, model, X_train, y_train, X_test, y_test
    ))

# --------------------------------------------------
# 4. Existing preprocessing strategy experiments
#    Run on the preprocessed dataset
# --------------------------------------------------

df = processed_df
X = df.drop(columns=["price"])
y = df["price"].astype(float)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

# Baseline
baseline = make_pipeline(X_train)
results.append(evaluate(
    "Preprocessed Baseline",
    baseline,
    X_train, y_train, X_test, y_test
))

# IQR filtering on training rows only
train_for_iqr = X_train.copy()
train_for_iqr["_target_price"] = y_train

keep = pd.Series(True, index=train_for_iqr.index)

for col in ["year", "kilometer", "_target_price"]:
    if col not in train_for_iqr.columns:
        continue

    values = train_for_iqr[col].dropna()

    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    keep &= (
        train_for_iqr[col].isna()
        | train_for_iqr[col].between(lower, upper)
    )

X_train_iqr = X_train.loc[keep]
y_train_iqr = y_train.loc[keep]

print(
    f"\nIQR filter retained {len(X_train_iqr)} "
    f"of {len(X_train)} training rows."
)

iqr_model = make_pipeline(X_train_iqr)
results.append(evaluate(
    "IQR Training Filter",
    iqr_model,
    X_train_iqr, y_train_iqr, X_test, y_test
))

# Log-transformed target
log_model = make_pipeline(X_train)
log_model.fit(X_train, np.log1p(y_train))

log_predictions = np.expm1(log_model.predict(X_test))
log_predictions = np.maximum(log_predictions, 0)

results.append({
    "Experiment": "Log-Transformed Target",
    "MAE": mean_absolute_error(y_test, log_predictions),
    "RMSE": np.sqrt(mean_squared_error(y_test, log_predictions)),
    "R2": r2_score(y_test, log_predictions)
})

print("\n--- Log-Transformed Target ---")
print(f"MAE:  Rs. {results[-1]['MAE']:,.2f}")
print(f"RMSE: Rs. {results[-1]['RMSE']:,.2f}")
print(f"R2:   {results[-1]['R2']:.4f}")

# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(RESULTS_DIR, "preprocessing_comparison.csv"),
    index=False
)

# Separate concise table for the required Phase 2 comparison
phase2_df = results_df[
    results_df["Experiment"].isin(
        ["Raw Data", "Preprocessed Data"]
    )
].copy()

phase2_df.to_csv(
    os.path.join(RESULTS_DIR, "raw_vs_preprocessed.csv"),
    index=False
)

print("\n===== ALL EXPERIMENTS =====")
print(results_df.to_string(index=False))

print("\n===== PHASE 2 RAW VS PREPROCESSED =====")
print(phase2_df.to_string(index=False))

# --------------------------------------------------
# 6. Generate comparison charts
# --------------------------------------------------

for metric in ["MAE", "RMSE", "R2"]:
    plt.figure(figsize=(10, 5))
    plt.bar(results_df["Experiment"], results_df[metric])
    plt.ylabel(metric)
    plt.title(f"Model and Preprocessing Comparison: {metric}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR, f"preprocessing_comparison_{metric.lower()}.png"
        ),
        dpi=150
    )
    plt.close()

for metric in ["MAE", "RMSE", "R2"]:
    plt.figure(figsize=(8, 5))
    plt.bar(phase2_df["Experiment"], phase2_df[metric])
    plt.ylabel(metric)
    plt.title(f"Raw vs Preprocessed Data: {metric}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR, f"raw_vs_preprocessed_{metric.lower()}.png"
        ),
        dpi=150
    )
    plt.close()

print("\nSaved results/preprocessing_comparison.csv")
print("Saved results/raw_vs_preprocessed.csv")
print("Saved comparison graphs in the graphs folder.")