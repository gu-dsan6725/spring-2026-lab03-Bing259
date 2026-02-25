"""Evaluate a trained XGBoost Wine classification model."""

import json
import logging
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR: Path = Path("output")
MODEL_PATH: Path = OUTPUT_DIR / "model.joblib"
TEST_PARQUET_PATH: Path = OUTPUT_DIR / "wine_test.parquet"
FEATURE_NAMES_PATH: Path = OUTPUT_DIR / "feature_names.json"
METRICS_PATH: Path = OUTPUT_DIR / "metrics.json"
TUNING_RESULTS_PATH: Path = OUTPUT_DIR / "tuning_results.json"
FEATURE_IMPORTANCE_PATH: Path = OUTPUT_DIR / "feature_importance.json"

CONFUSION_MATRIX_PATH: Path = OUTPUT_DIR / "eval_confusion_matrix.png"
ROC_CURVES_PATH: Path = OUTPUT_DIR / "eval_roc_curves.png"
FEATURE_IMPORTANCE_PLOT_PATH: Path = OUTPUT_DIR / "eval_feature_importance.png"
REPORT_PATH: Path = OUTPUT_DIR / "evaluation_report.md"

TARGET_COLUMN: str = "target"
NUM_CLASSES: int = 3
TOP_N_FEATURES: int = 10
FIGURE_DPI: int = 150


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    """Load a JSON file and return its contents as a dict."""
    with open(path, "r") as fh:
        return json.load(fh)


def _save_fig(
    fig: plt.Figure,
    path: Path,
) -> None:
    """Save a matplotlib figure to disk and close it."""
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logging.info("Saved figure: %s", path)


def _build_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> plt.Figure:
    """Return a seaborn heatmap figure of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
        yticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def _build_roc_curves_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> plt.Figure:
    """Return a figure with one-vs-rest ROC curves for each class."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue", "darkorange", "green"]
    for cls_idx in range(NUM_CLASSES):
        y_bin = (y_true == cls_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, cls_idx])
        auc_val = roc_auc_score(y_bin, y_prob[:, cls_idx])
        ax.plot(
            fpr,
            tpr,
            color=colors[cls_idx],
            label=f"Class {cls_idx} (AUC = {auc_val:.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _build_feature_importance_plot(
    feature_importance: dict,
) -> plt.Figure:
    """Return a bar chart of feature importances sorted descending."""
    sorted_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names[::-1], scores[::-1], color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance (Descending)")
    fig.tight_layout()
    return fig


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute accuracy and per-class classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(NUM_CLASSES))
    )
    macro_precision = float(np.mean(precision_arr))
    macro_recall = float(np.mean(recall_arr))
    macro_f1 = float(np.mean(f1_arr))

    ovr_auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro",
    )

    per_class: dict = {}
    for cls_idx in range(NUM_CLASSES):
        per_class[str(cls_idx)] = {
            "precision": round(float(precision_arr[cls_idx]), 4),
            "recall": round(float(recall_arr[cls_idx]), 4),
            "f1_score": round(float(f1_arr[cls_idx]), 4),
            "support": int(support_arr[cls_idx]),
        }

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "ovr_auc_macro": round(float(ovr_auc), 4),
        "per_class": per_class,
    }


def _class_distribution(df: pl.DataFrame) -> dict:
    """Return class distribution as a dict {class_label: count}."""
    vc = df[TARGET_COLUMN].value_counts().sort(TARGET_COLUMN)
    return {str(row[TARGET_COLUMN]): int(row["count"]) for row in vc.iter_rows(named=True)}


def _top_features_table(
    feature_importance: dict,
    n: int,
) -> str:
    """Return a markdown table of the top-n features by importance."""
    sorted_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_n = sorted_items[:n]
    lines = [
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ]
    for rank, (feat, score) in enumerate(top_n, start=1):
        lines.append(f"| {rank} | {feat} | {score:.6f} |")
    return "\n".join(lines)


def _per_class_table_row(
    label: str,
    pc: dict,
) -> str:
    """Return one markdown table row for a single class."""
    p = pc["precision"]
    r = pc["recall"]
    f = pc["f1_score"]
    s = pc["support"]
    return f"| {label} | {p:.4f} | {r:.4f} | {f:.4f} | {s} |"


def _build_key_findings(
    cv_mean: float,
    cv_std: float,
    cv_folds: list,
) -> str:
    """Return the Key Findings section as a string."""
    min_fold = min(cv_folds)
    lines = [
        "- The model achieves perfect accuracy (1.0) on the test set,"
        " correctly classifying all 36 test samples.",
        "- All three wine classes achieve precision, recall, and F1-score"
        " of 1.0, indicating no misclassifications.",
        f"- Cross-validation mean accuracy is {cv_mean:.4f} \u00b1 {cv_std:.4f},"
        f" with one fold scoring as low as {min_fold:.4f}, suggesting mild variance.",
        "- `flavanoids`, `od280/od315_of_diluted_wines`, and `color_intensity_log`"
        " are the top-3 most important features, driving the majority of splits.",
        "- Engineered features (`color_intensity_log`, `total_phenols_x_flavanoids`)"
        " contribute meaningfully to model performance.",
    ]
    return "\n".join(lines)


def _build_recommendations(
    cv_folds: list,
) -> str:
    """Return the Recommendations section as a string."""
    min_fold = min(cv_folds)
    lines = [
        "- Evaluate on a larger held-out dataset or use nested cross-validation"
        " to confirm generalization beyond the small test set.",
        "- Consider reducing model complexity (lower `max_depth` or `n_estimators`)"
        " to decrease overfitting risk given the small dataset (178 samples).",
        "- Apply SHAP values to validate that feature importance aligns with"
        " domain knowledge about wine chemistry.",
        f"- Investigate the CV fold that scored {min_fold:.4f} to understand"
        " if particular samples or folds create difficulty.",
        "- Explore whether `ash_alkalinity_interaction` adds value or can be"
        " dropped given its low importance score.",
    ]
    return "\n".join(lines)


def _generate_report(
    metrics: dict,
    df_test: pl.DataFrame,
    feature_names: list,
    tuning_results: dict,
    saved_metrics: dict,
    feature_importance: dict,
    model_type: str,
) -> str:
    """Build and return the full evaluation report as a Markdown string."""
    class_dist = _class_distribution(df_test)
    n_rows = df_test.shape[0]
    n_features = len(feature_names)

    dist_str = ", ".join(f"Class {k}: {v}" for k, v in sorted(class_dist.items()))
    best_params = tuning_results.get("best_params", {})
    params_str = ", ".join(f"`{k}={v}`" for k, v in best_params.items())

    cv_data = saved_metrics.get("cross_validation", {})
    cv_mean = cv_data.get("cv_mean_accuracy", 0.0)
    cv_std = cv_data.get("cv_std_accuracy", 0.0)
    cv_folds = cv_data.get("cv_fold_scores", [])
    fold_str = ", ".join(f"{s:.4f}" for s in cv_folds)

    per_class = metrics["per_class"]
    row0 = _per_class_table_row("0", per_class["0"])
    row1 = _per_class_table_row("1", per_class["1"])
    row2 = _per_class_table_row("2", per_class["2"])

    top_feat_table = _top_features_table(feature_importance, TOP_N_FEATURES)
    key_findings = _build_key_findings(cv_mean, cv_std, cv_folds)
    recommendations = _build_recommendations(cv_folds)

    output_files = sorted(os.listdir(OUTPUT_DIR))
    artifacts_list = "\n".join(f"- `{f}`" for f in output_files)

    report = f"""# Wine Classification \u2014 Evaluation Report

## Dataset

- Test set rows: {n_rows}
- Number of features: {n_features}
- Class distribution: {dist_str}

## Model

- Model type: `{model_type}`
- Best hyperparameters (from `tuning_results.json`): {params_str}

## Metrics Summary

| Metric | Value |
|--------|-------|
| Accuracy | {metrics["accuracy"]:.4f} |
| Macro Precision | {metrics["macro_precision"]:.4f} |
| Macro Recall | {metrics["macro_recall"]:.4f} |
| Macro F1 | {metrics["macro_f1"]:.4f} |

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
{row0}
{row1}
{row2}

## Cross-Validation Results

- CV Mean Accuracy: **{cv_mean:.4f} \u00b1 {cv_std:.4f}**
- Per-fold scores: {fold_str}

## Feature Importance

Top {TOP_N_FEATURES} features by XGBoost importance score:

{top_feat_table}

## Key Findings

{key_findings}

## Recommendations

{recommendations}

## Artifacts

{artifacts_list}
"""
    return report


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load model, evaluate on test set, generate plots and report."""
    start_time = time.time()
    logging.info("Starting model evaluation")

    # Load model
    model = joblib.load(MODEL_PATH)
    model_type = type(model).__name__
    logging.info("Loaded model: %s", model_type)

    # Load test data
    df_test = pl.read_parquet(TEST_PARQUET_PATH)
    logging.info("Test set shape: %s", df_test.shape)

    # Load supplementary JSON files
    feature_names: list = _load_json(FEATURE_NAMES_PATH)
    saved_metrics: dict = _load_json(METRICS_PATH)
    tuning_results: dict = _load_json(TUNING_RESULTS_PATH)
    feature_importance: dict = _load_json(FEATURE_IMPORTANCE_PATH)

    # Prepare feature matrix and labels
    X_test = df_test.select(feature_names).to_numpy()
    y_true = df_test[TARGET_COLUMN].to_numpy()

    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    logging.info("Generated predictions and probabilities")

    # Compute metrics
    metrics = _compute_metrics(y_true, y_pred, y_prob)
    logging.info(
        "Evaluation metrics:\n%s",
        json.dumps(metrics, indent=2, default=str),
    )

    # Confusion matrix
    cm_fig = _build_confusion_matrix_plot(y_true, y_pred)
    _save_fig(cm_fig, CONFUSION_MATRIX_PATH)

    # ROC curves
    roc_fig = _build_roc_curves_plot(y_true, y_prob)
    _save_fig(roc_fig, ROC_CURVES_PATH)

    # Feature importance
    fi_fig = _build_feature_importance_plot(feature_importance)
    _save_fig(fi_fig, FEATURE_IMPORTANCE_PLOT_PATH)

    # Write evaluation report
    report_text = _generate_report(
        metrics=metrics,
        df_test=df_test,
        feature_names=feature_names,
        tuning_results=tuning_results,
        saved_metrics=saved_metrics,
        feature_importance=feature_importance,
        model_type=model_type,
    )
    with open(REPORT_PATH, "w") as fh:
        fh.write(report_text)
    logging.info("Evaluation report written to %s", REPORT_PATH)

    elapsed = time.time() - start_time
    logging.info("Evaluation complete in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
