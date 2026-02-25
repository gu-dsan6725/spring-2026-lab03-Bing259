"""Exploratory Data Analysis on the UCI Wine dataset.

Loads the dataset, computes summary statistics, generates distribution
plots, creates a correlation heatmap, checks class balance, and
identifies outliers using the IQR method.
"""

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output"
FIGURE_DPI: int = 150
IQR_MULTIPLIER: float = 1.5
TARGET_COLUMN: str = "target"


def _ensure_output_dir(
    output_dir: str,
) -> Path:
    """Create the output directory if it does not exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_dataset() -> pl.DataFrame:
    """Load the UCI Wine dataset and return as a polars DataFrame."""
    wine = load_wine(as_frame=False)
    feature_names = list(wine.feature_names)
    data = wine.data
    target = wine.target

    df = pl.DataFrame({name: data[:, i] for i, name in enumerate(feature_names)})
    df = df.with_columns(pl.Series(TARGET_COLUMN, target))

    logger.info(f"Loaded Wine dataset with shape: {df.shape}")
    return df


def _compute_summary_statistics(
    df: pl.DataFrame,
) -> dict:
    """Compute summary statistics for all feature columns."""
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    stats = {}
    for col in feature_cols:
        col_data = df[col]
        stats[col] = {
            "mean": round(float(col_data.mean()), 4),
            "std": round(float(col_data.std()), 4),
            "min": round(float(col_data.min()), 4),
            "max": round(float(col_data.max()), 4),
            "null_count": int(col_data.null_count()),
        }

    logger.info(f"Summary statistics:\n{json.dumps(stats, indent=2, default=str)}")
    return stats


def _plot_distributions(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Generate histogram distribution plots for each feature."""
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        values = df[col].to_list()
        axes[i].hist(values, bins=30, edgecolor="black", alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    filepath = output_path / "distributions.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Distribution plots saved to {filepath}")


def _plot_correlation_matrix(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Generate a correlation matrix heatmap for all feature columns."""
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    corr_data = {}
    for col in feature_cols:
        correlations = []
        for other_col in feature_cols:
            corr_value = df.select(pl.corr(col, other_col)).item()
            correlations.append(round(float(corr_value), 3))
        corr_data[col] = correlations

    corr_matrix = pl.DataFrame(corr_data).to_numpy()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=feature_cols,
        yticklabels=feature_cols,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    filepath = output_path / "correlation_matrix.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Correlation matrix saved to {filepath}")


def _plot_class_balance(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Generate a bar chart showing the distribution of wine classes."""
    class_counts = df.group_by(TARGET_COLUMN).len().sort(TARGET_COLUMN)
    classes = [str(v) for v in class_counts[TARGET_COLUMN].to_list()]
    counts = class_counts["len"].to_list()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(classes, counts, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Wine Class Distribution")
    for idx, cnt in enumerate(counts):
        ax.text(idx, cnt + 0.3, str(cnt), ha="center", va="bottom")

    plt.tight_layout()
    filepath = output_path / "class_balance.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()

    balance_info = dict(zip(classes, counts))
    logger.info(f"Class distribution:\n{json.dumps(balance_info, indent=2, default=str)}")
    logger.info(f"Class balance plot saved to {filepath}")


def _identify_outliers(
    df: pl.DataFrame,
) -> dict:
    """Identify outliers using the IQR method for all feature columns."""
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    outlier_counts = {}
    for col in feature_cols:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        count = df.filter((pl.col(col) < lower) | (pl.col(col) > upper)).height
        outlier_counts[col] = count

    logger.info(
        "Outlier counts (IQR method):\n%s",
        json.dumps(outlier_counts, indent=2, default=str),
    )
    return outlier_counts


def run_eda() -> None:
    """Run the full exploratory data analysis pipeline."""
    start_time = time.time()
    logger.info("Starting EDA on the UCI Wine dataset...")

    output_path = _ensure_output_dir(OUTPUT_DIR)
    df = _load_dataset()
    _compute_summary_statistics(df)
    _plot_distributions(df, output_path)
    _plot_correlation_matrix(df, output_path)
    _plot_class_balance(df, output_path)
    _identify_outliers(df)

    df.write_parquet(output_path / "wine_raw.parquet")
    logger.info(f"Raw dataset saved to {output_path / 'wine_raw.parquet'}")

    elapsed = time.time() - start_time
    logger.info(f"EDA completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    run_eda()
