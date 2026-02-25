import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output")


def _load_data(
    train_path: Path,
    test_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load train and test parquet files."""
    logging.info("Loading engineered datasets")
    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)
    return train_df, test_df


def _tune_hyperparameters(
    X_train: pl.DataFrame,
    y_train: pl.Series,
) -> dict:
    """Tune XGBoost hyperparameters with RandomizedSearchCV."""
    logging.info("Tuning hyperparameters with RandomizedSearchCV")
    xgb = XGBClassifier(eval_metric="mlogloss", random_state=42)

    param_distributions = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train.to_pandas(), y_train.to_pandas())

    best_params = random_search.best_params_
    logging.info(f"Best parameters found:\n{json.dumps(best_params, indent=2, default=str)}")

    tuning_results_path = OUTPUT_DIR / "tuning_results.json"
    with open(tuning_results_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    logging.info(f"Saved tuning results to {tuning_results_path}")
    return best_params


def _train_model(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    best_params: dict,
) -> XGBClassifier:
    """Train final XGBoost model using the best hyperparameters."""
    logging.info("Training final XGBoost model")

    # Check class balance automatically with objective
    model = XGBClassifier(**best_params, eval_metric="mlogloss", random_state=42)
    model.fit(X_train.to_pandas(), y_train.to_pandas())

    # Save the model
    model_path = OUTPUT_DIR / "xgboost_model.joblib"
    joblib.dump(model, model_path)
    logging.info(f"Saved trained model to {model_path}")

    return model


def _evaluate_model(
    model: XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.Series,
) -> None:
    """Evaluate model and save metric logs and visualizations."""
    logging.info("Evaluating model on test set")

    y_pred = model.predict(X_test.to_pandas())
    y_true = y_test.to_pandas()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
    }

    logging.info(f"Evaluation metrics:\n{json.dumps(metrics, indent=2, default=str)}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    logging.info(f"Saved confusion matrix to {cm_path}")

    # Feature Importances
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=X_test.columns, color="b")
    plt.title("XGBoost Feature Importances")
    plt.tight_layout()
    fi_path = OUTPUT_DIR / "feature_importances.png"
    plt.savefig(fi_path)
    plt.close()
    logging.info(f"Saved feature importances to {fi_path}")

    # Write evaluation report
    report_path = OUTPUT_DIR / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write("## Overview\n")
        f.write("This report provides evaluation metrics for the XGBoost wine classifier.\n\n")
        f.write("## Metrics\n")
        for k, v in metrics.items():
            f.write(f"- **{k.capitalize()}**: {v:.4f}\n")
        f.write("\n## Artifacts\n")
        f.write("- [Confusion Matrix](confusion_matrix.png)\n")
        f.write("- [Feature Importances](feature_importances.png)\n")

    logging.info(f"Saved evaluation report to {report_path}")


def main() -> None:
    """Run model training and evaluation."""
    train_path = OUTPUT_DIR / "wine_train.parquet"
    test_path = OUTPUT_DIR / "wine_test.parquet"

    if not train_path.exists() or not test_path.exists():
        logging.error("Engineered data files not found in output/")
        return

    train_df, test_df = _load_data(train_path, test_path)

    X_train = train_df.drop("target")
    y_train = train_df["target"]
    X_test = test_df.drop("target")
    y_test = test_df["target"]

    best_params = _tune_hyperparameters(X_train, y_train)
    model = _train_model(X_train, y_train, best_params)
    _evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
