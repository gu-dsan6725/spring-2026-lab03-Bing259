# Wine Classification Pipeline Plan

## Overview
This plan outlines the steps to build a Wine classification pipeline using `scikit-learn`'s `load_wine` dataset, `polars` for data manipulation, and `xgboost` for classification. All scripts will adhere strictly to the project's coding standards.

## Directory Structure
- Scripts will be stored in: `part2_antigravity/src/`
- Outputs (plots, reports) will be stored in: `part2_antigravity/output/`

## Step 1: Exploratory Data Analysis (EDA) - `src/01_eda.py`
- **Data Loading**: Retrieve the dataset using `sklearn.datasets.load_wine()`.
- **Data Conversion**: Convert feature arrays and targets into a `polars` DataFrame for manipulation.
- **Profiling**: Calculate summary statistics (mean, median, standard deviation) for each feature.
- **Visualizations**: Generate feature distributions and a correlation heatmap.
- **Output**: Save EDA plots to the `output/` directory.

## Step 2: Feature Engineering & Preprocessing - `src/02_feature_engineering.py`
- **Data Splitting**: Divide the data into training and testing sets.
- **Scaling**: Apply feature scaling using `StandardScaler` to ensure features have zero mean and unit variance.
- **Type Annotations**: Ensure all functions use proper typing, with one parameter per line.
- **Output**: Save the processed train and test datasets for model training.

## Step 3: Model Training, Evaluation & Reporting - `src/03_xgboost_model.py`
- **Algorithm**: Initialize an XGBoost classifier.
- **Hyperparameter Tuning**: Perform hyperparameter tuning using `RandomizedSearchCV` with 20 iterations and 5-fold stratified cross-validation.
- **Output (Tuning)**: Save the tuning results to `output/tuning_results.json`.
- **Training**: Train the final model on the training set using the best hyperparameters.
- **Inference**: Generate predictions using the test set.
- **Metrics**: Compute accuracy, precision, recall, and F1-score.
- **Logging**: Pretty-print evaluation metrics dictionaries in log messages using `json.dumps(data, indent=2, default=str)`.
- **Visualizations**: 
  - Plot a Confusion Matrix.
  - Plot XGBoost Feature Importances.
- **Output**: Save evaluation plots to the `output/` directory.

## General Coding Standards Checklist
- [ ] Use Python 3.11+.
- [ ] Manage dependencies with `uv`.
- [ ] Ensure all private functions are prefixed with `_` and placed at the top of the file, followed by public functions.
- [ ] Keep function length between 30-50 lines with two blank lines between definitions.
- [ ] Use multi-line imports.
- [ ] Avoid hard-coded constants inside test/functional logic; use top-level typed constants instead.
- [ ] Include standard logging configuration in every script.
- [ ] Run `uv run ruff check --fix <filename>` and `uv run python -m py_compile <filename>` after writing each script.
