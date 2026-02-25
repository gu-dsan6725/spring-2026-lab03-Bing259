import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output")


def _load_wine_data() -> pl.DataFrame:
    """Load the wine dataset and return as a Polars DataFrame."""
    logging.info("Loading wine dataset")
    data = load_wine()

    df = pl.DataFrame(data.data, schema=data.feature_names)
    df = df.with_columns(pl.Series("target", data.target))

    return df


def _profile_data(
    df: pl.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute summary statistics for each feature."""
    logging.info("Profiling data")
    stats_dict = {}
    for col in df.columns:
        if col == "target":
            continue
        col_series = df[col]
        # Calculate IQR for outliers
        q1 = col_series.quantile(0.25)
        q3 = col_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = col_series.filter(
            (col_series < lower_bound) | (col_series > upper_bound)
        ).len()

        stats_dict[col] = {
            "mean": float(col_series.mean()),
            "median": float(col_series.median()),
            "std": float(col_series.std()),
            "min": float(col_series.min()),
            "max": float(col_series.max()),
            "missing": int(col_series.null_count()),
            "outliers": int(outliers_count),
        }

    logging.info(f"Data profile:\n{json.dumps(stats_dict, indent=2, default=str)}")

    # Check class balance
    logging.info("Checking class balance of target variable")
    target_counts = df["target"].value_counts().sort("target").to_dicts()
    logging.info(f"Class balance:\n{json.dumps(target_counts, indent=2, default=str)}")

    # Check overall missing values
    total_missing = df.null_count().sum_horizontal()[0]
    logging.info(f"Total missing values in dataset: {total_missing}")

    return stats_dict


def _plot_distributions(
    df: pl.DataFrame,
) -> None:
    """Generate and save feature distributions plot."""
    logging.info("Generating feature distributions plot")
    features = [col for col in df.columns if col != "target"]

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.histplot(df[feature].to_list(), ax=axes[i], kde=True)
        axes[i].set_title(feature)

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "feature_distributions.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved feature distributions to {plot_path}")


def _plot_correlation_heatmap(
    df: pl.DataFrame,
) -> None:
    """Generate and save correlation heatmap."""
    logging.info("Generating correlation heatmap")
    features = [col for col in df.columns if col != "target"]
    df_features = df.select(features)

    # Calculate correlation matrix using polars to pandas since seaborn prefers pandas
    corr_matrix = df_features.to_pandas().corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved correlation heatmap to {plot_path}")


def main() -> None:
    """Run exploratory data analysis pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_wine_data()
    _profile_data(df)
    _plot_distributions(df)
    _plot_correlation_heatmap(df)

    # Save raw data for feature engineering
    raw_data_path = OUTPUT_DIR / "wine_raw.parquet"
    df.write_parquet(raw_data_path)
    logging.info(f"Saved raw data to {raw_data_path}")


if __name__ == "__main__":
    main()
