# ML Lab: Six Experiments on One Generated Dataset

## What this project does
This project generates one synthetic customer dataset and reuses it for:
1. Linear and polynomial regression to predict Monthly_Spending.
2. Multiple linear regression to predict Annual_Income.
3. Decision Tree classification of Income_Class.
4. Random Forest classification of Income_Class.
5. Gradient Boosting classification of Income_Class.
6. Hierarchical clustering of customer purchasing behaviour.

**Academic note:** This is a generated dataset, not the Boston Housing, Titanic, Adult Census Income, or Wholesale Customers dataset. If the instructor requires the named datasets, use those original datasets instead. Results from synthetic data are demonstrations, not evidence of real-world model performance.

## Setup in VS Code (Windows)
1. Install Python 3.10+ and VS Code's Python extension.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:


python -m pip install pandas scikit-learn matplotlib

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt terminal and run:
   ```bat
   .venv\Scripts\activate.bat
   ```

6. Install libraries:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Generate the shared dataset:
   ```powershell
   python generate_dataset.py
   ```

8. Run each experiment separately:
   ```powershell
   python exp1_regression_monthly_spending.py
   python exp2_regression_annual_income.py
   python exp3_decision_tree_income_class.py
   python exp4_random_forest_income_class.py
   python exp5_gradient_boosting_income_class.py
   python exp6_hierarchical_clustering.py
   ```

## Output files
- `customer_ml_dataset.csv`: shared generated dataset.
- `outputs/`: figures, predictions, feature importances and cluster profiles.
- Figures are saved as PNG at 300 dpi and also displayed when each script runs.

## Why these methods?
- Regression: target is numerical.
- Decision Tree, Random Forest, Gradient Boosting: target is categorical (`Income_Class`).
- Hierarchical clustering: unsupervised grouping, no target column used.

## Interpretation tips
- MAE and RMSE: lower is better.
- R²: closer to 1 is generally better; can be negative if predictions are poor.
- Accuracy: fraction of correct classifications; use precision/recall/F1 too.
- Confusion matrix: rows are actual classes and columns are predicted classes.
- Dendrogram: visual summary of how observations/clusters merge.
- Ward linkage uses Euclidean distance and works best after scaling the features.
