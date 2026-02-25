## Plan: Wine Classification Pipeline

### Objective

Build a complete ML pipeline for classifying wines into 3 classes using the UCI Wine dataset
(`sklearn.datasets.load_wine()`). The pipeline covers EDA, feature engineering, and XGBoost
classification with hyperparameter tuning and 5-fold cross-validation. All scripts live in
`part1_claude_code/src/` and write artifacts to `output/`.

---

### Steps

#### Step 1 — EDA (`part1_claude_code/src/01_eda.py`)

**What it does:**
- Loads `load_wine()` and converts to a Polars DataFrame (178 rows × 13 features + target)
- Computes summary statistics (mean, std, min, max, null counts) for every column
- Generates per-feature distribution histograms saved as `output/distributions.png`
- Generates a full correlation heatmap saved as `output/correlation_matrix.png`
- Reports class balance (3 classes: 0, 1, 2) and saves a bar chart `output/class_balance.png`
- Detects outliers via IQR (1.5× rule) per feature and logs counts
- Saves the raw dataset as `output/wine_raw.parquet` for use in Step 2

**CLAUDE.md requirements met:**
- Polars for all data wrangling
- Logging with prescribed `basicConfig` format
- Constants at file top (`OUTPUT_DIR`, `FIGURE_DPI`, `IQR_MULTIPLIER`)
- Private helper functions (`_load_dataset`, `_compute_summary_statistics`,
  `_plot_distributions`, `_plot_correlation`, `_plot_class_balance`, `_detect_outliers`,
  `_ensure_output_dir`) above `main()`
- Type annotations on every parameter

---

#### Step 2 — Feature Engineering (`part1_claude_code/src/02_feature_engineering.py`)

**What it does:**
- Reads `output/wine_raw.parquet`
- Creates **≥ 3 derived features**:
  1. `alcohol_to_malic_ratio` = `alcohol / malic_acid`
  2. `total_phenols_x_flavanoids` = `total_phenols * flavanoids`
  3. `color_intensity_log` = `log1p(color_intensity)`
  4. `ash_alkalinity_interaction` = `ash * alcalinity_of_ash`
- Applies `sklearn.preprocessing.StandardScaler` to all numeric features (fit on train only)
- Performs a stratified 80/20 train/test split (`random_state=42`)
- Saves:
  - `output/wine_train.parquet`
  - `output/wine_test.parquet`
  - `output/scaler.joblib` (fitted scaler for reproducibility)
  - `output/feature_names.json` (list of final feature columns)

**Dependencies:** Step 1 must produce `output/wine_raw.parquet`

---

#### Step 3 — XGBoost Model (`part1_claude_code/src/03_xgboost_model.py`)

**What it does:**
- Reads `output/wine_train.parquet` and `output/wine_test.parquet`
- Runs **hyperparameter tuning** with `RandomizedSearchCV`:
  - 20 iterations, 5-fold `StratifiedKFold`, scoring=`accuracy`
  - Search space: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`
  - Saves best params and all iteration scores to `output/tuning_results.json`
- Trains a final `XGBClassifier` with the best found params (`objective="multi:softprob"`, `num_class=3`)
- Runs **5-fold stratified cross-validation** on the tuned model and logs per-fold accuracy
- Evaluates on held-out test set:
  - Accuracy, macro Precision, Recall, F1-score
  - Per-class precision/recall breakdown
  - Confusion matrix saved as `output/confusion_matrix.png`
  - Feature importance bar chart saved as `output/feature_importance.png`
- Saves:
  - `output/model.joblib`
  - `output/metrics.json` (CV scores + test metrics)
  - `output/feature_importance.json`
  - `output/tuning_results.json` (best params + all RandomizedSearchCV iteration scores)

**Dependencies:** Step 2 must produce train/test parquet files

---

### Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Data manipulation | `polars` | Mandated by CLAUDE.md |
| ML framework | `xgboost.XGBClassifier` | Specified in task; natively supports multi-class |
| Hyperparameter tuning | `RandomizedSearchCV` (20 iter, 5-fold) | Efficient search over large param space |
| Cross-validation | `StratifiedKFold(n_splits=5)` | Preserves class balance across folds |
| Scaler | `StandardScaler` | Simple, effective; fit on train only to prevent leakage |
| Serialization | `joblib` for model/scaler, `parquet` for data | Consistent with demo reference |
| Plot format | `matplotlib` + `seaborn`, DPI=150 | Same as demo; readable output |
| Random seed | `RANDOM_STATE = 42` | Reproducibility constant at file top |
| File naming | `01_`, `02_`, `03_` prefix | Run-order clarity |

**Trade-offs considered:**
- `polars` has no native `StratifiedKFold`—numpy arrays will be used for sklearn CV, then results
  converted back to polars for logging/saving.
- Derived features are created before scaling; the scaler is fit only on training data to prevent
  data leakage.

---

### Testing Strategy

After each script is written, `uv run ruff check --fix` and `uv run python -m py_compile` run
automatically via hooks. End-to-end verification:

```bash
uv run python part1_claude_code/src/01_eda.py
uv run python part1_claude_code/src/02_feature_engineering.py
uv run python part1_claude_code/src/03_xgboost_model.py
```

Success criteria:
- No Python or ruff errors
- `output/` contains all expected files (see below)
- Test-set accuracy ≥ 0.90 (Wine dataset is well-separated; XGBoost should achieve this easily)
- Logging output uses the exact prescribed format

---

### Expected Output

```
output/
├── wine_raw.parquet          # Raw dataset from Step 1
├── distributions.png         # Per-feature histograms
├── correlation_matrix.png    # Feature correlation heatmap
├── class_balance.png         # Class distribution bar chart
├── wine_train.parquet        # Scaled training set from Step 2
├── wine_test.parquet         # Scaled test set from Step 2
├── scaler.joblib             # Fitted StandardScaler
├── feature_names.json        # Final feature list
├── model.joblib              # Trained XGBoost model
├── metrics.json              # CV + test metrics
├── feature_importance.json   # Feature importance scores
├── tuning_results.json       # RandomizedSearchCV results
├── confusion_matrix.png      # Test-set confusion matrix
└── feature_importance.png    # Feature importance bar chart
```

**Scripts produced:**
```
part1_claude_code/src/
├── 01_eda.py
├── 02_feature_engineering.py
└── 03_xgboost_model.py
```
