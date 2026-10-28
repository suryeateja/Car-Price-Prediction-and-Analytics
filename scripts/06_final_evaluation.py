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

# --------------------------------------------------
# 1. Paths and folders
# --------------------------------------------------

DATA = "data/processed/car_details_v4_cleaned.csv"

os.makedirs("results", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# --------------------------------------------------
# 2. Load and prepare the dataset
# --------------------------------------------------

df = pd.read_csv(DATA)

# Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

# Remove rows without a valid target
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])
df = df[df["price"] > 0].reset_index(drop=True)

X = df.drop(columns=["price"])
y = df["price"].astype(float)

# Identify numerical and categorical features
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# --------------------------------------------------
# 3. Preprocessing pipeline
# --------------------------------------------------

preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        num_cols
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        cat_cols
    )
])

# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# 5. Train the selected model
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
    )
])

model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Predictions and evaluation metrics
# --------------------------------------------------

pred = model.predict(X_test)
residuals = y_test.to_numpy() - pred

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nFINAL MODEL EVALUATION")
print("=" * 45)
print(f"Dataset rows: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: Rs. {mae:,.2f}")
print(f"RMSE: Rs. {rmse:,.2f}")

# --------------------------------------------------
# 7. Actual vs Predicted plot
# --------------------------------------------------

plt.figure(figsize=(7, 6))

plt.scatter(y_test, pred, alpha=0.6)

low = min(y_test.min(), pred.min())
high = max(y_test.max(), pred.max())

plt.plot(
    [low, high],
    [low, high],
    linestyle="--"
)

plt.xlabel("Actual Price (Rs.)")
plt.ylabel("Predicted Price (Rs.)")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()

plt.savefig(
    "graphs/actual_vs_predicted.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 8. Residual analysis plot
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Price (Rs.)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Analysis - Gradient Boosting")
plt.tight_layout()

plt.savefig(
    "graphs/residual_analysis.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 9. Save individual predictions
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "Actual_Price": y_test.to_numpy(),
    "Predicted_Price": pred,
    "Residual": residuals
})

predictions_df.to_csv(
    "results/test_predictions.csv",
    index=False
)

# --------------------------------------------------
# 10. Save final evaluation report
# --------------------------------------------------

with open(
    "results/final_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("CAR PRICE PREDICTION AND ANALYTICS\n")
    f.write("=" * 45 + "\n")
    f.write("Dataset: car_details_v4.csv\n")
    f.write(f"Rows used: {len(df)}\n")
    f.write(f"Features: {X.shape[1]}\n")
    f.write(f"Training rows: {len(X_train)}\n")
    f.write(f"Testing rows: {len(X_test)}\n")
    f.write("Train-test split: 80/20\n")
    f.write("Random state: 42\n")
    f.write("Selected model: Gradient Boosting Regressor\n")
    f.write("Number of estimators: 100\n\n")

    f.write("EVALUATION METRICS\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write(f"MAE: Rs. {mae:,.2f}\n")
    f.write(f"RMSE: Rs. {rmse:,.2f}\n\n")

    f.write("PREPROCESSING\n")
    f.write("Numerical missing values: median imputation\n")
    f.write("Categorical missing values: most-frequent imputation\n")
    f.write("Categorical encoding: One-Hot Encoding\n\n")

    f.write("EXPERIMENT FINDINGS\n")
    f.write(
        "Preprocessing modestly improved performance "
        "in the earlier model comparison.\n"
    )
    f.write(
        "IQR-based training-row filtering performed worse "
        "on the evaluated test set.\n"
    )
    f.write(
        "Log-target transformation reduced MAE but worsened "
        "RMSE and R2 in the earlier experiment.\n"
    )

    f.write("\nLIMITATIONS\n")
    f.write(
        "Evaluation uses a single holdout split. "
        "Further validation is recommended.\n"
    )

print("\nAll outputs saved successfully:")
print("1. graphs/actual_vs_predicted.png")
print("2. graphs/residual_analysis.png")
print("3. results/test_predictions.csv")
print("4. results/final_summary.txt")
