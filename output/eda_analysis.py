import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
DATA_PATH: str = "output/wine_raw.parquet"
OUTPUT_DIR: Path = Path("output")
DIST_PLOT_PATH: Path = OUTPUT_DIR / "eda_skill_distributions.png"
CORR_PLOT_PATH: Path = OUTPUT_DIR / "eda_skill_correlation.png"
TARGET_COL: str = "target"
IQR_MULTIPLIER: float = 1.5
HIST_BINS: int = 20
FIGSIZE_DIST: tuple[int, int] = (18, 14)
FIGSIZE_CORR: tuple[int, int] = (14, 12)


def _compute_summary_stats(
    df: pl.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Compute mean, median, std, min, max for each numeric feature."""
    stats: dict = {}
    for col in feature_cols:
        series = df[col]
        stats[col] = {
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
        }
    return stats


def _check_missing_values(
    df: pl.DataFrame,
) -> dict:
    """Count and compute percentage of missing values per column."""
    total_rows = df.height
    missing: dict = {}
    for col in df.columns:
        null_count = df[col].null_count()
        pct = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        missing[col] = {"count": null_count, "percentage": pct}
    return missing


def _check_duplicates(df: pl.DataFrame) -> int:
    """Return the number of duplicate rows in the DataFrame."""
    return df.height - df.unique().height


def _detect_outliers_iqr(
    df: pl.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Identify outlier counts per feature using the IQR method."""
    outlier_counts: dict = {}
    for col in feature_cols:
        series = df[col]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        count = int(((series < lower) | (series > upper)).sum())
        outlier_counts[col] = count
    return outlier_counts


def _plot_distributions(
    df: pl.DataFrame,
    feature_cols: list[str],
    output_path: Path,
) -> None:
    """Plot histograms for each numeric feature and save to output_path."""
    n_cols = 4
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGSIZE_DIST)
    axes_flat = axes.flatten()

    for idx, col in enumerate(feature_cols):
        ax = axes_flat[idx]
        values = df[col].to_list()
        ax.hist(values, bins=HIST_BINS, edgecolor="black", color="steelblue", alpha=0.75)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)

    for idx in range(len(feature_cols), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Feature Distributions — UCI Wine Dataset", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Distribution plot saved to %s", output_path)


def _plot_correlation_heatmap(
    df: pl.DataFrame,
    all_cols: list[str],
    output_path: Path,
) -> None:
    """Plot a seaborn correlation matrix heatmap and save to output_path."""
    corr_matrix = df.select(all_cols).to_pandas().corr()
    fig, ax = plt.subplots(figsize=FIGSIZE_CORR)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title("Correlation Matrix — UCI Wine Dataset", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Correlation heatmap saved to %s", output_path)


def load_data(data_path: str) -> pl.DataFrame:
    """Load parquet data from data_path and return a polars DataFrame."""
    logging.info("Loading data from %s", data_path)
    df = pl.read_parquet(data_path)
    logging.info("Loaded dataset with shape: %s rows x %s cols", df.height, df.width)
    return df


def run_summary_stats(
    df: pl.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Compute and log summary statistics for all features."""
    logging.info("Computing summary statistics...")
    stats = _compute_summary_stats(df, feature_cols)
    logging.info(
        "Summary statistics:\n%s",
        json.dumps(stats, indent=2, default=str),
    )
    return stats


def run_missing_values(df: pl.DataFrame) -> dict:
    """Check and log missing values per column."""
    logging.info("Checking for missing values...")
    missing = _check_missing_values(df)
    logging.info(
        "Missing values per column:\n%s",
        json.dumps(missing, indent=2, default=str),
    )
    return missing


def run_duplicate_check(df: pl.DataFrame) -> int:
    """Check and log duplicate row count."""
    logging.info("Checking for duplicate rows...")
    n_dupes = _check_duplicates(df)
    logging.info("Duplicate rows: %d", n_dupes)
    return n_dupes


def run_outlier_detection(
    df: pl.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Detect and log outlier counts per feature using IQR method."""
    logging.info("Detecting outliers using IQR method (multiplier=%.1f)...", IQR_MULTIPLIER)
    outliers = _detect_outliers_iqr(df, feature_cols)
    logging.info(
        "Outlier counts per feature:\n%s",
        json.dumps(outliers, indent=2, default=str),
    )
    return outliers


def run_plots(
    df: pl.DataFrame,
    feature_cols: list[str],
    all_cols: list[str],
) -> None:
    """Generate and save distribution and correlation plots."""
    logging.info("Generating distribution plots...")
    _plot_distributions(df, feature_cols, DIST_PLOT_PATH)

    logging.info("Generating correlation heatmap...")
    _plot_correlation_heatmap(df, all_cols, CORR_PLOT_PATH)


def log_summary(
    df: pl.DataFrame,
    missing: dict,
    n_dupes: int,
    outliers: dict,
) -> None:
    """Log a high-level summary of key EDA findings."""
    total_missing = sum(v["count"] for v in missing.values())
    total_outliers = sum(outliers.values())
    summary = {
        "dataset_shape": {"rows": df.height, "columns": df.width},
        "total_missing_values": total_missing,
        "duplicate_rows": n_dupes,
        "total_outliers_across_features": total_outliers,
        "features_with_outliers": {k: v for k, v in outliers.items() if v > 0},
    }
    logging.info(
        "KEY FINDINGS SUMMARY:\n%s",
        json.dumps(summary, indent=2, default=str),
    )


def main() -> None:
    """Run the full EDA pipeline on the UCI Wine dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    all_cols = feature_cols + [TARGET_COL]

    run_summary_stats(df, feature_cols)
    missing = run_missing_values(df)
    n_dupes = run_duplicate_check(df)
    outliers = run_outlier_detection(df, feature_cols)
    run_plots(df, feature_cols, all_cols)
    log_summary(df, missing, n_dupes, outliers)

    logging.info("EDA complete.")


if __name__ == "__main__":
    main()
