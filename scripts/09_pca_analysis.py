import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load the original dataset
df = pd.read_csv("data/raw/car_details_v4.csv")

print("Original dataset shape:", df.shape)

# 2. Select numerical input features
X = df.select_dtypes(include=["number"]).drop(
    columns=["Price"], errors="ignore"
)

print("Numerical features shape:", X.shape)

# 3. Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# 4. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 5. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 6. Calculate explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

print("\nExplained variance by each component:")
for i, variance in enumerate(explained_variance, start=1):
    print(f"PC{i}: {variance:.4f} ({variance * 100:.2f}%)")

# 7. Find components needed to retain 95% variance
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)

print("\nNumber of original numerical features:", X.shape[1])
print("Components needed to retain 95% variance:",
      pca_95.n_components_)
print("Shape after PCA:", X_pca_95.shape)

# 8. Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)
plt.axhline(y=0.95, linestyle="--", label="95% variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/pca_explained_variance.png")
plt.show()