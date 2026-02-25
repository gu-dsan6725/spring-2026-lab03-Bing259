# Model Evaluation Report

## Executive Summary

An XGBoost classifier (`XGBClassifier`) was trained on the UCI Wine dataset to distinguish 3 wine
varietal classes from 17 features (13 original + 4 engineered). After hyperparameter tuning with
RandomizedSearchCV (20 iterations, 5-fold stratified CV), the best model achieved **97.93% mean
CV accuracy** on the training set and **100% accuracy** on the 36-sample held-out test set, with
perfect precision, recall, and F1-score across all three classes.

## Dataset Overview

| Property | Value |
|----------|-------|
| Total samples | 178 |
| Training samples | 142 (80%, stratified) |
| Test samples | 36 (20%, stratified) |
| Number of features | 17 (13 original + 4 engineered) |
| Target variable | `target` — wine varietal class (0, 1, 2) |

## Model Configuration

| Hyperparameter | Value |
|----------------|-------|
| Model type | `XGBClassifier` |
| `objective` | `multi:softprob` |
| `num_class` | 3 |
| `n_estimators` | 200 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.6 |
| `gamma` | 0.1 |
| Tuning method | `RandomizedSearchCV` (20 iter, 5-fold stratified CV) |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 1.0000 |
| Macro Precision | 1.0000 |
| Macro Recall | 1.0000 |
| Macro F1-Score | 1.0000 |
| OVR AUC (macro) | 1.0000 |
| CV Mean Accuracy (5-fold) | 0.9793 |
| CV Std Accuracy | 0.0414 |

## Feature Importance (Top 5)

| Rank | Feature | Importance Score |
|------|---------|-----------------|
| 1 | `flavanoids` | 0.1376 |
| 2 | `od280/od315_of_diluted_wines` | 0.1263 |
| 3 | `color_intensity_log` *(engineered)* | 0.1076 |
| 4 | `color_intensity` | 0.1069 |
| 5 | `proline` | 0.0940 |

## Recommendations for Improvement

1. **Validate with nested CV:** A perfect test score on 36 samples may be optimistic. Use
   nested cross-validation (outer 10-fold, inner 5-fold) on the full 178-sample dataset to
   produce a more reliable accuracy estimate and detect potential overfitting.

2. **Reduce model depth:** The rank-2 tuning candidate achieved near-identical CV accuracy
   (97.91%) with `max_depth=4` instead of 6. Shallower trees are less likely to overfit on
   a 142-sample training set and generalise better to unseen data.

3. **Drop near-zero importance features:** `ash_alkalinity_interaction` (0.0049) and
   `nonflavanoid_phenols` (0.0076) contribute almost nothing to the model's splits. Removing
   them simplifies the feature space and may reduce noise without hurting accuracy.
