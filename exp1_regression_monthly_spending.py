"""
Experiment 1: Regression to Predict Monthly Spending

Steps:
1. Generate the customer dataset.
2. Load the generated dataset.
3. Apply Simple Linear Regression.
4. Apply Polynomial Regression.
5. Evaluate both models.
6. Save predictions and visualizations.

Dataset size: 1500 customers
Target variable: Monthly_Spending
Input feature: Annual_Income
"""

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

# Importing this module automatically executes generate_dataset.py.
# It generates customer_ml_dataset.csv before the experiment starts.
import generate_dataset

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# STEP 2: SET PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "customer_ml_dataset.csv"

OUTPUT_DIR = BASE_DIR / "outputs"

# Create outputs folder automatically if it does not exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 3: LOAD THE GENERATED DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("\n" + "=" * 65)
print("EXPERIMENT 1: REGRESSION TO PREDICT MONTHLY SPENDING")
print("=" * 65)

print("\nDataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Total customers: {len(df)}")

print("\nFirst five rows:")
print(df.head())


# ============================================================
# STEP 4: SELECT INPUT AND TARGET VARIABLES
# ============================================================

# Input feature (X): Annual Income
# Target variable (y): Monthly Spending

X = df[["Annual_Income"]]
y = df["Monthly_Spending"]

print("\nInput feature: Annual_Income")
print("Target variable: Monthly_Spending")


# ============================================================
# STEP 5: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining and testing split completed.")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# ============================================================
# STEP 6: APPLY SIMPLE LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

# Train the model.
linear_model.fit(X_train, y_train)

# Predict monthly spending for test records.
linear_pred = linear_model.predict(X_test)

print("\n" + "-" * 65)
print("MODEL 1: SIMPLE LINEAR REGRESSION")
print("-" * 65)

print("\nRegression equation:")

print(
    f"Monthly Spending = {linear_model.intercept_:.3f} "
    f"+ ({linear_model.coef_[0]:.6f} * Annual Income)"
)


# ============================================================
# STEP 7: APPLY POLYNOMIAL REGRESSION
# ============================================================

# Degree 2 allows the model to learn a curved relationship.
poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

# Train the polynomial regression model.
poly_model.fit(X_train, y_train)

# Predict monthly spending.
poly_pred = poly_model.predict(X_test)

print("\n" + "-" * 65)
print("MODEL 2: POLYNOMIAL REGRESSION")
print("-" * 65)

print("Polynomial degree: 2")
print("The model can learn a curved relationship between income and spending.")


# ============================================================
# STEP 8: EVALUATE BOTH MODELS
# ============================================================

def evaluate_model(actual, predicted, model_name):
    """
    Calculate regression evaluation metrics.
    """

    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = np.sqrt(mse)

    r2 = r2_score(actual, predicted)

    print(f"\n{model_name}")
    print("-" * 45)
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2_Score": r2
    }


print("\n" + "=" * 65)
print("MODEL EVALUATION RESULTS")
print("=" * 65)

linear_results = evaluate_model(
    y_test,
    linear_pred,
    "Simple Linear Regression"
)

poly_results = evaluate_model(
    y_test,
    poly_pred,
    "Polynomial Regression (Degree 2)"
)


# ============================================================
# STEP 9: COMPARE MODEL PERFORMANCE
# ============================================================

results_df = pd.DataFrame([
    linear_results,
    poly_results
])

print("\n" + "=" * 65)
print("COMPARISON OF REGRESSION MODELS")
print("=" * 65)

print(results_df.round(4).to_string(index=False))

# Save evaluation results.
results_df.to_csv(
    OUTPUT_DIR / "exp1_model_comparison.csv",
    index=False
)


# ============================================================
# STEP 10: SAVE ACTUAL AND PREDICTED VALUES
# ============================================================

comparison = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Linear_Predicted": linear_pred,
    "Polynomial_Predicted": poly_pred
})

comparison.to_csv(
    OUTPUT_DIR / "exp1_predictions.csv",
    index=False
)

print("\nSample predictions:")
print(comparison.head(10).round(2).to_string(index=False))

print("\nPredictions saved to:")
print(OUTPUT_DIR / "exp1_predictions.csv")


# ============================================================
# STEP 11: PLOT ACTUAL VALUES AND REGRESSION LINES
# ============================================================

# Sort test records by annual income for a clear regression plot.
plot_df = pd.DataFrame({
    "Annual_Income": X_test["Annual_Income"].to_numpy(),
    "Actual": y_test.to_numpy()
})

plot_df = plot_df.sort_values("Annual_Income")

# Generate predictions in sorted order.
linear_sorted = linear_model.predict(
    plot_df[["Annual_Income"]]
)

poly_sorted = poly_model.predict(
    plot_df[["Annual_Income"]]
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=plot_df,
    x="Annual_Income",
    y="Actual",
    alpha=0.45,
    label="Actual Spending"
)

plt.plot(
    plot_df["Annual_Income"],
    linear_sorted,
    label="Linear Regression"
)

plt.plot(
    plot_df["Annual_Income"],
    poly_sorted,
    label="Polynomial Regression"
)

plt.title("Experiment 1: Predicting Monthly Spending")
plt.xlabel("Annual Income")
plt.ylabel("Monthly Spending")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "exp1_regression_fit.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nRegression comparison graph saved.")


# ============================================================
# STEP 12: ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    linear_pred,
    alpha=0.6,
    label="Linear Regression"
)

# Draw the ideal prediction line.
min_value = min(y_test.min(), linear_pred.min())
max_value = max(y_test.max(), linear_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Ideal Prediction"
)

plt.xlabel("Actual Monthly Spending")
plt.ylabel("Predicted Monthly Spending")

plt.title(
    "Experiment 1: Actual vs Predicted Monthly Spending"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "exp1_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Actual vs predicted graph saved.")


# ============================================================
# STEP 13: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT 1 COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nDataset:")
print(f"Total customers: {len(df)}")

print("\nModels applied:")
print("1. Simple Linear Regression")
print("2. Polynomial Regression (Degree 2)")

print("\nTarget variable:")
print("Monthly_Spending")

print("\nInput feature:")
print("Annual_Income")

print("\nGenerated output files:")
print("1. exp1_predictions.csv")
print("2. exp1_model_comparison.csv")
print("3. exp1_regression_fit.png")
print("4. exp1_actual_vs_predicted.png")

print(f"\nAll outputs are saved in:\n{OUTPUT_DIR}")

print("\nExperiment 1 finished!")