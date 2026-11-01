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
Experiment 2: Apply multiple linear regression to generated customer data.
Target: Annual_Income
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from common import load_dataset, OUTPUT_DIR, save_plot, print_regression_metrics

df = load_dataset()

target = "Annual_Income"
features = [
    "Age", "Education_Years", "Work_Experience", "Family_Size",
    "City_Category", "Employment_Type", "Membership_Type"
]
X = df[features]
y = df[target]

numeric_features = ["Age", "Education_Years", "Work_Experience", "Family_Size"]
categorical_features = ["City_Category", "Employment_Type", "Membership_Type"]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nEXPERIMENT 2: Multiple Linear Regression to predict Annual Income")
print_regression_metrics(y_test, pred, "Multiple Linear Regression")

results = pd.DataFrame({"Actual_Income": y_test.to_numpy(), "Predicted_Income": pred})
results.to_csv(OUTPUT_DIR / "exp2_predictions.csv", index=False)
print("\nSample predictions:")
print(results.head(10).round(2))

plt.figure(figsize=(7, 5))
plt.scatter(y_test, pred, alpha=0.55)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, linestyle="--")
plt.xlabel("Actual Annual Income")
plt.ylabel("Predicted Annual Income")
plt.title("Experiment 2: Actual vs Predicted Annual Income")
save_plot("exp2_actual_vs_predicted.png")

residuals = y_test.to_numpy() - pred
plt.figure(figsize=(7, 5))
sns.scatterplot(x=pred, y=residuals, alpha=0.55)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Annual Income")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Experiment 2: Residual Plot")
save_plot("exp2_residual_plot.png")
