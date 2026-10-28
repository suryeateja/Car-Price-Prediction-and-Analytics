import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. Paths and folders
# --------------------------------------------------

DATA = "data/processed/car_details_v4_cleaned.csv"

os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA)

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])
df = df[df["price"] > 0].reset_index(drop=True)

X = df.drop(columns=["price"])
y = df["price"].astype(float)

num_cols = X.select_dtypes(include="number").columns.tolist()

cat_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Dataset shape:", df.shape)
print("Numerical features:", len(num_cols))
print("Categorical features:", len(cat_cols))

# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# 4. Numerical preprocessing WITHOUT PCA
# --------------------------------------------------

num_without_pca = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# --------------------------------------------------
# 5. Numerical preprocessing WITH PCA
# --------------------------------------------------

num_with_pca = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95))
])

# --------------------------------------------------
# 6. Categorical preprocessing
# --------------------------------------------------

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# --------------------------------------------------
# 7. Create both preprocessing pipelines
# --------------------------------------------------

preprocessor_without_pca = ColumnTransformer([
    ("num", num_without_pca, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

preprocessor_with_pca = ColumnTransformer([
    ("num", num_with_pca, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# --------------------------------------------------
# 8. Create both Gradient Boosting models
# --------------------------------------------------

model_without_pca = Pipeline([
    ("preprocessor", preprocessor_without_pca),
    ("regressor", GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    ))
])

model_with_pca = Pipeline([
    ("preprocessor", preprocessor_with_pca),
    ("regressor", GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# --------------------------------------------------
# 9. Train and evaluate WITHOUT PCA
# --------------------------------------------------

print("\nTraining model WITHOUT PCA...")

model_without_pca.fit(X_train, y_train)

pred_without_pca = model_without_pca.predict(X_test)

r2_without = r2_score(y_test, pred_without_pca)
mae_without = mean_absolute_error(y_test, pred_without_pca)
rmse_without = np.sqrt(
    mean_squared_error(y_test, pred_without_pca)
)

# --------------------------------------------------
# 10. Train and evaluate WITH PCA
# --------------------------------------------------

print("Training model WITH PCA...")

model_with_pca.fit(X_train, y_train)

pred_with_pca = model_with_pca.predict(X_test)

r2_with = r2_score(y_test, pred_with_pca)
mae_with = mean_absolute_error(y_test, pred_with_pca)
rmse_with = np.sqrt(
    mean_squared_error(y_test, pred_with_pca)
)

# --------------------------------------------------
# 11. Display comparison
# --------------------------------------------------

print("\n" + "=" * 55)
print("GRADIENT BOOSTING: PCA vs WITHOUT PCA")
print("=" * 55)

print("\nWITHOUT PCA")
print(f"R2 Score: {r2_without:.4f}")
print(f"MAE: Rs. {mae_without:,.2f}")
print(f"RMSE: Rs. {rmse_without:,.2f}")

print("\nWITH PCA")
print(f"R2 Score: {r2_with:.4f}")
print(f"MAE: Rs. {mae_with:,.2f}")
print(f"RMSE: Rs. {rmse_with:,.2f}")

# --------------------------------------------------
# 12. Display PCA component count
# --------------------------------------------------

fitted_preprocessor = model_with_pca.named_steps["preprocessor"]

fitted_pca = (
    fitted_preprocessor
    .named_transformers_["num"]
    .named_steps["pca"]
)

print("\nPCA ANALYSIS")
print("Original numerical features:", len(num_cols))
print("Principal components retained:", fitted_pca.n_components_)
print(
    "Variance retained:",
    f"{fitted_pca.explained_variance_ratio_.sum() * 100:.2f}%"
)

# --------------------------------------------------
# 13. Save comparison results
# --------------------------------------------------

results_df = pd.DataFrame({
    "Model": ["Gradient Boosting without PCA",
              "Gradient Boosting with PCA"],
    "R2_Score": [r2_without, r2_with],
    "MAE": [mae_without, mae_with],
    "RMSE": [rmse_without, rmse_with]
})

results_df.to_csv(
    "results/pca_model_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("results/pca_model_comparison.csv")