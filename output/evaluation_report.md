# Wine Classification — Evaluation Report

## Dataset

- Test set rows: 36
- Number of features: 17
- Class distribution: Class 0: 12, Class 1: 14, Class 2: 10

## Model

- Model type: `XGBClassifier`
- Best hyperparameters (from `tuning_results.json`): `subsample=1.0`, `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `gamma=0.1`, `colsample_bytree=0.6`

## Metrics Summary

| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 |
| Macro Precision | 1.0000 |
| Macro Recall | 1.0000 |
| Macro F1 | 1.0000 |

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 1.0000 | 1.0000 | 1.0000 | 12 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 14 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 10 |

## Cross-Validation Results

- CV Mean Accuracy: **0.9793 ± 0.0414**
- Per-fold scores: 0.8966, 1.0000, 1.0000, 1.0000, 1.0000

## Feature Importance

Top 10 features by XGBoost importance score:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | flavanoids | 0.137577 |
| 2 | od280/od315_of_diluted_wines | 0.126325 |
| 3 | color_intensity_log | 0.107572 |
| 4 | color_intensity | 0.106876 |
| 5 | proline | 0.094018 |
| 6 | total_phenols | 0.083956 |
| 7 | total_phenols_x_flavanoids | 0.070239 |
| 8 | magnesium | 0.047158 |
| 9 | hue | 0.039794 |
| 10 | proanthocyanins | 0.039308 |

## Key Findings

- The model achieves perfect accuracy (1.0) on the test set, correctly classifying all 36 test samples.
- All three wine classes achieve precision, recall, and F1-score of 1.0, indicating no misclassifications.
- Cross-validation mean accuracy is 0.9793 ± 0.0414, with one fold scoring as low as 0.8966, suggesting mild variance.
- `flavanoids`, `od280/od315_of_diluted_wines`, and `color_intensity_log` are the top-3 most important features, driving the majority of splits.
- Engineered features (`color_intensity_log`, `total_phenols_x_flavanoids`) contribute meaningfully to model performance.

## Recommendations

- Evaluate on a larger held-out dataset or use nested cross-validation to confirm generalization beyond the small test set.
- Consider reducing model complexity (lower `max_depth` or `n_estimators`) to decrease overfitting risk given the small dataset (178 samples).
- Apply SHAP values to validate that feature importance aligns with domain knowledge about wine chemistry.
- Investigate the CV fold that scored 0.8966 to understand if particular samples or folds create difficulty.
- Explore whether `ash_alkalinity_interaction` adds value or can be dropped given its low importance score.

## Artifacts

- `__pycache__`
- `class_balance.png`
- `confusion_matrix.png`
- `correlation_matrix.png`
- `distributions.png`
- `eda_analysis.py`
- `eda_skill_correlation.png`
- `eda_skill_distributions.png`
- `eval_confusion_matrix.png`
- `eval_feature_importance.png`
- `eval_roc_curves.png`
- `evaluate_model.py`
- `feature_importance.json`
- `feature_importance.png`
- `feature_names.json`
- `metrics.json`
- `model.joblib`
- `scaler.joblib`
- `tuning_results.json`
- `wine_raw.parquet`
- `wine_test.parquet`
- `wine_train.parquet`
