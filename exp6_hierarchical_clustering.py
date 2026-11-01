import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible synthetic customer dataset
RANDOM_STATE = 42
N_CUSTOMERS = 1500  # Changed from 3000 to 1500
rng = np.random.default_rng(RANDOM_STATE)

age = rng.integers(18, 71, N_CUSTOMERS)
education_years = np.clip(np.round(rng.normal(14, 2.5, N_CUSTOMERS)), 8, 21).astype(int)
work_experience = np.clip(age - 18 - education_years * 0.45 + rng.normal(0, 4, N_CUSTOMERS), 0, 45).round(1)
family_size = rng.integers(1, 7, N_CUSTOMERS)

gender = rng.choice(["Female", "Male", "Other"], N_CUSTOMERS, p=[0.49, 0.49, 0.02])
city_category = rng.choice(["Tier 1", "Tier 2", "Tier 3"], N_CUSTOMERS, p=[0.35, 0.40, 0.25])

employment_type = rng.choice(
    ["Salaried", "Self-employed", "Business", "Student", "Unemployed"],
    N_CUSTOMERS, p=[0.48, 0.18, 0.15, 0.11, 0.08]
)

membership_type = rng.choice(
    ["Basic", "Silver", "Gold", "Platinum"],
    N_CUSTOMERS, p=[0.42, 0.30, 0.20, 0.08]
)

purchase_category = rng.choice(
    ["Groceries", "Electronics", "Clothing", "Home", "Travel"],
    N_CUSTOMERS, p=[0.30, 0.20, 0.22, 0.18, 0.10]
)

city_factor = pd.Series(city_category).map({
    "Tier 1": 1.25, "Tier 2": 1.0, "Tier 3": 0.82
}).to_numpy()

employment_factor = pd.Series(employment_type).map({
    "Salaried": 1.0,
    "Self-employed": 1.18,
    "Business": 1.45,
    "Student": 0.32,
    "Unemployed": 0.25
}).to_numpy()

income_base = (
    14000 + education_years * 2600 + work_experience * 1050
    + age * 180 + rng.normal(0, 9000, N_CUSTOMERS)
)

annual_income = np.maximum(
    9000, income_base * city_factor * employment_factor
).round(0)

online_purchases = np.clip(
    rng.poisson(np.maximum(1, annual_income / 22000 + 3)), 0, 60
)

store_visits = np.clip(
    rng.poisson(5 + family_size * 1.4), 0, 35
)

website_time = np.clip(
    rng.normal(18 + online_purchases * 1.3, 8, N_CUSTOMERS),
    1, 120
).round(1)

discount_usage = np.clip(
    rng.poisson(2.5 + online_purchases * 0.12), 0, 25
)

membership_bonus = pd.Series(membership_type).map({
    "Basic": 0,
    "Silver": 350,
    "Gold": 800,
    "Platinum": 1500
}).to_numpy()

monthly_spending = (
    900 + annual_income * 0.018 + online_purchases * 95
    + store_visits * 45 + website_time * 8 + membership_bonus
    + family_size * 120 + rng.normal(0, 650, N_CUSTOMERS)
)

monthly_spending = np.maximum(250, monthly_spending).round(2)

savings = np.maximum(
    0,
    annual_income * rng.uniform(0.04, 0.30, N_CUSTOMERS)
    - monthly_spending * rng.uniform(0.5, 3.0, N_CUSTOMERS)
).round(2)

credit_score = np.clip(
    520 + annual_income / 1100 + work_experience * 1.2
    - (employment_type == "Unemployed") * 35
    + rng.normal(0, 35, N_CUSTOMERS),
    300, 850
).round().astype(int)

customer_satisfaction = np.clip(
    rng.normal(6.2, 1.7, N_CUSTOMERS)
    + np.isin(membership_type, ["Gold", "Platinum"]) * 0.6,
    1, 10
).round(1)

# Calculate customer spending score
spending_score = (
    0.45 * (monthly_spending / max(monthly_spending.max(), 1) * 100)
    + 0.25 * (online_purchases / max(online_purchases.max(), 1) * 100)
    + 0.15 * (website_time / 120 * 100)
    + 0.15 * (customer_satisfaction / 10 * 100)
).round(2)

# Create income classes
q1, q2 = np.quantile(annual_income, [0.50, 0.80])

income_class = np.where(
    annual_income < q1, "Low",
    np.where(annual_income < q2, "Medium", "High")
)

# Create DataFrame
df = pd.DataFrame({
    "Customer_ID": np.arange(1, N_CUSTOMERS + 1),
    "Age": age,
    "Education_Years": education_years,
    "Work_Experience": work_experience,
    "Annual_Income": annual_income,
    "Monthly_Spending": monthly_spending,
    "Savings": savings,
    "Credit_Score": credit_score,
    "Online_Purchases": online_purchases,
    "Store_Visits": store_visits,
    "Website_Time_Minutes": website_time,
    "Discount_Usage": discount_usage,
    "Family_Size": family_size,
    "Customer_Satisfaction": customer_satisfaction,
    "Gender": gender,
    "City_Category": city_category,
    "Employment_Type": employment_type,
    "Membership_Type": membership_type,
    "Purchase_Category": purchase_category,
    "Income_Class": income_class,
    "Spending_Score": spending_score,
})

# Save dataset
out_path = Path(__file__).resolve().parent / "customer_ml_dataset.csv"
df.to_csv(out_path, index=False)

print(f"Dataset generated successfully: {out_path}")
print(f"Shape: {df.shape}")

print("\nFirst five rows:")
print(df.head())

print("\nIncome class distribution:")
print(df["Income_Class"].value_counts())

print("\nSummary statistics:")
print(df.select_dtypes(include="number").describe().round(2))




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
from common import load_dataset, OUTPUT_DIR, save_plot

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
