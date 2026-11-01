
"""
Experiment 3: Decision Tree classifier on generated customer data.
Target: Income_Class (Low, Medium, High)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
    ("classifier", DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=8, random_state=42, class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nEXPERIMENT 3: Decision Tree Classification")
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))
print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, pred, labels=model.named_steps["classifier"].classes_))

labels = model.named_steps["classifier"].classes_
cm = confusion_matrix(y_test, pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format="d")
plt.title("Experiment 3: Decision Tree Confusion Matrix")
save_plot("exp3_decision_tree_confusion_matrix.png")

# Visualise the first part of the tree after preprocessing
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
tree = model.named_steps["classifier"]
plt.figure(figsize=(22, 10))
plot_tree(
    tree, feature_names=feature_names, class_names=list(tree.classes_),
    filled=True, rounded=True, max_depth=3, fontsize=7
)
plt.title("Experiment 3: Decision Tree (display limited to depth 3)")
save_plot("exp3_decision_tree_plot.png")

pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": pred}).to_csv(
    OUTPUT_DIR / "exp3_predictions.csv", index=False
)
