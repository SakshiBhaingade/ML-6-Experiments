

"""
Experiment 6: Hierarchical Agglomerative Clustering.
Distance metric: Euclidean distance
Linkage criterion: Ward linkage
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_PATH = BASE_DIR / "customer_ml_dataset.csv"

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. First run: python generate_dataset.py"
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nMissing values by column:")
    print(df.isnull().sum())
    return df

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def print_regression_metrics(y_true, y_pred, label="Model"):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{label} performance")
    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2_score(y_true, y_pred):.3f}")

df = load_dataset()

features = [
    "Annual_Income", "Monthly_Spending", "Savings",
    "Online_Purchases", "Store_Visits", "Website_Time_Minutes",
    "Customer_Satisfaction", "Spending_Score"
]
X = df[features].copy()
X = X.fillna(X.median(numeric_only=True))

# Scaling is essential because income/spending have much larger numeric ranges.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dendrogram can become very crowded for thousands of points.
# Plot a reproducible random sample for readability, while clustering all rows below.
rng = __import__("numpy").random.default_rng(42)
sample_size = min(250, len(df))
sample_indices = rng.choice(len(df), size=sample_size, replace=False)
Z = linkage(X_scaled[sample_indices], method="ward", metric="euclidean")

plt.figure(figsize=(14, 6))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=45, leaf_font_size=9)
plt.title("Experiment 6: Hierarchical Clustering Dendrogram (sample)")
plt.xlabel("Cluster or sample index")
plt.ylabel("Ward distance")
save_plot("exp6_dendrogram.png")

# Ward linkage is defined with Euclidean distance.
n_clusters = 4
clusterer = AgglomerativeClustering(
    n_clusters=n_clusters, linkage="ward", metric="euclidean"
)
cluster_labels = clusterer.fit_predict(X_scaled)

result = df[["Customer_ID"] + features].copy()
result["Cluster"] = cluster_labels
result.to_csv(OUTPUT_DIR / "exp6_customer_clusters.csv", index=False)

print("\nEXPERIMENT 6: Hierarchical Clustering")
print("Distance metric: Euclidean")
print("Linkage criterion: Ward")
print(f"Number of clusters: {n_clusters}")
print("\nCustomers per cluster:")
print(result["Cluster"].value_counts().sort_index())
print("\nCluster profile (mean values):")
profile = result.groupby("Cluster")[features].mean().round(2)
print(profile)
profile.to_csv(OUTPUT_DIR / "exp6_cluster_profile.csv")

# PCA projects the scaled data into 2D for visualisation only.
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
plot_df = pd.DataFrame({
    "PC1": X_2d[:, 0], "PC2": X_2d[:, 1],
    "Cluster": cluster_labels.astype(str)
})
plt.figure(figsize=(9, 6))
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Cluster", alpha=0.65, s=35)
plt.title("Experiment 6: Customer Clusters (PCA visualisation)")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(title="Cluster")
save_plot("exp6_clusters_pca.png")

# Simple feature distribution visualisation by cluster
plt.figure(figsize=(9, 5))
sns.boxplot(data=result, x="Cluster", y="Monthly_Spending")
plt.title("Experiment 6: Monthly Spending by Cluster")
save_plot("exp6_spending_by_cluster.png")
