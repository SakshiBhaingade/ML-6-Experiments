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
