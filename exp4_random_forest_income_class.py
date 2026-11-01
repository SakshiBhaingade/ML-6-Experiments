
"""
Experiment 4: Random Forest classifier on generated customer data.
Uses the same target/features as Experiment 3 for a fair comparison.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

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
target = "Income_Class"
features = [
    "Age", "Education_Years", "Work_Experience", "Monthly_Spending",
    "Savings", "Credit_Score", "Online_Purchases", "Store_Visits",
    "Website_Time_Minutes", "Discount_Usage", "Family_Size",
    "Customer_Satisfaction", "Gender", "City_Category", "Employment_Type",
    "Membership_Type", "Purchase_Category"
]
X, y = df[features], df[target]
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=250, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1, class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nEXPERIMENT 4: Random Forest Classification")
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))
labels = model.named_steps["classifier"].classes_
cm = confusion_matrix(y_test, pred, labels=labels)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
    cmap="Greens", values_format="d"
)
plt.title("Experiment 4: Random Forest Confusion Matrix")
save_plot("exp4_random_forest_confusion_matrix.png")

# Feature importance after one-hot encoding
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importances = model.named_steps["classifier"].feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names, "Importance": importances
}).sort_values("Importance", ascending=False).head(15)
importance_df.to_csv(OUTPUT_DIR / "exp4_feature_importance.csv", index=False)

plt.figure(figsize=(10, 7))
plt.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
plt.xlabel("Feature Importance")
plt.title("Experiment 4: Top 15 Random Forest Features")
save_plot("exp4_feature_importance.png")

pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": pred}).to_csv(
    OUTPUT_DIR / "exp4_predictions.csv", index=False
)
