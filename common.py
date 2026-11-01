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
