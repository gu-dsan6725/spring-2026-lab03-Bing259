"""Train and evaluate an XGBoost classifier on the UCI Wine dataset.

Loads prepared train/test splits, runs hyperparameter tuning with
RandomizedSearchCV (20 iterations, 5-fold stratified CV), trains the
best model, evaluates with cross-validation and on the held-out test
set, and saves all artifacts to the output directory.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output"
TARGET_COLUMN: str = "target"
FIGURE_DPI: int = 150
RANDOM_STATE: int = 42
CV_FOLDS: int = 5
N_ITER_SEARCH: int = 20
CV_SCORING: str = "accuracy"
NUM_CLASSES: int = 3

# Hyperparameter search space
PARAM_DISTRIBUTIONS: dict = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3, 0.5],
}


def _load_splits(
    output_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load train/test splits from parquet files and return numpy arrays."""
    path = Path(output_dir)
    train_df = pl.read_parquet(path / "wine_train.parquet")
    test_df = pl.read_parquet(path / "wine_test.parquet")

    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
    x_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df[TARGET_COLUMN].to_numpy()
    x_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df[TARGET_COLUMN].to_numpy()

    logger.info(f"Loaded splits — train: {x_train.shape}, test: {x_test.shape}")
    return x_train, x_test, y_train, y_test, feature_cols


def _run_hyperparameter_tuning(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[XGBClassifier, RandomizedSearchCV]:
    """Run RandomizedSearchCV with 20 iterations and 5-fold stratified CV."""
    logger.info(
        f"Starting hyperparameter tuning: {N_ITER_SEARCH} iterations, "
        f"{CV_FOLDS}-fold stratified CV..."
    )
    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring=CV_SCORING,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    logger.info(f"Best CV accuracy: {search.best_score_:.4f}")
    logger.info(f"Best params:\n{json.dumps(search.best_params_, indent=2, default=str)}")
    return search.best_estimator_, search


def _save_tuning_results(
    search: RandomizedSearchCV,
    output_path: Path,
) -> None:
    """Save all RandomizedSearchCV iteration scores and best params to JSON."""
    cv_results = search.cv_results_
    candidates = []
    for i in range(len(cv_results["params"])):
        params = {
            k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in cv_results["params"][i].items()
        }
        candidates.append(
            {
                "rank": int(cv_results["rank_test_score"][i]),
                "mean_accuracy": round(float(cv_results["mean_test_score"][i]), 4),
                "std_accuracy": round(float(cv_results["std_test_score"][i]), 4),
                "params": params,
            }
        )
    candidates.sort(key=lambda x: x["rank"])

    best_params = {
        k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
        for k, v in search.best_params_.items()
    }
    results = {
        "best_params": best_params,
        "best_cv_accuracy": round(float(search.best_score_), 4),
        "n_iterations": N_ITER_SEARCH,
        "cv_folds": CV_FOLDS,
        "all_candidates": candidates,
    }
    filepath = output_path / "tuning_results.json"
    filepath.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Tuning results saved to {filepath}")


def _run_cross_validation(
    model: XGBClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Run 5-fold stratified CV on the tuned model and log per-fold accuracy."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, x_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)

    cv_results = {
        "cv_mean_accuracy": round(float(np.mean(scores)), 4),
        "cv_std_accuracy": round(float(np.std(scores)), 4),
        "cv_fold_scores": [round(float(s), 4) for s in scores],
    }
    logger.info(
        f"CV accuracy: {cv_results['cv_mean_accuracy']} (+/- {cv_results['cv_std_accuracy']})"
    )
    logger.info(f"Per-fold scores: {cv_results['cv_fold_scores']}")
    return cv_results


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute accuracy, macro precision/recall/F1, and per-class breakdown."""
    report = classification_report(y_true, y_pred, output_dict=True)
    per_class = {
        str(cls): {
            "precision": round(report[str(cls)]["precision"], 4),
            "recall": round(report[str(cls)]["recall"], 4),
            "f1_score": round(report[str(cls)]["f1-score"], 4),
            "support": int(report[str(cls)]["support"]),
        }
        for cls in sorted(set(y_true))
    }
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro")), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro")), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": per_class,
    }
    logger.info(f"Test set metrics:\n{json.dumps(metrics, indent=2, default=str)}")
    return metrics


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save a confusion matrix heatmap to the output directory."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Class 0", "Class 1", "Class 2"],
        yticklabels=["Class 0", "Class 1", "Class 2"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Wine Classification")
    plt.tight_layout()
    filepath = output_path / "confusion_matrix.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Confusion matrix saved to {filepath}")


def _plot_feature_importance(
    model: XGBClassifier,
    feature_cols: list[str],
    output_path: Path,
) -> dict:
    """Save a feature importance bar chart and return a scores dictionary."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = [feature_cols[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(importances)), importances[sorted_idx], alpha=0.8, edgecolor="black")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.set_title("XGBoost Feature Importance — Wine Classification")
    plt.tight_layout()
    filepath = output_path / "feature_importance.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Feature importance plot saved to {filepath}")

    return {feature_cols[i]: round(float(importances[i]), 6) for i in range(len(feature_cols))}


def run_training() -> None:
    """Run the full XGBoost training and evaluation pipeline."""
    start_time = time.time()
    logger.info("Starting XGBoost classification pipeline for UCI Wine dataset...")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test, feature_cols = _load_splits(OUTPUT_DIR)

    model, search = _run_hyperparameter_tuning(x_train, y_train)
    _save_tuning_results(search, output_path)

    cv_results = _run_cross_validation(model, x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = _compute_metrics(y_test, y_pred)
    metrics["cross_validation"] = cv_results

    _plot_confusion_matrix(y_test, y_pred, output_path)
    importance_dict = _plot_feature_importance(model, feature_cols, output_path)

    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (output_path / "feature_importance.json").write_text(
        json.dumps(importance_dict, indent=2, default=str)
    )
    joblib.dump(model, output_path / "model.joblib")
    logger.info("Saved model.joblib, metrics.json, and feature_importance.json")

    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    run_training()
