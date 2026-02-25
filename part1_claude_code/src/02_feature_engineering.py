"""Feature engineering for the UCI Wine dataset.

Creates 4 derived features from raw wine measurements, performs a
stratified 80/20 train/test split, scales features using StandardScaler
fit on the training set only, and saves all outputs for model training.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
INPUT_PATH: str = "output/wine_raw.parquet"
OUTPUT_DIR: str = "output"
TARGET_COLUMN: str = "target"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42


def _ensure_output_dir(
    output_dir: str,
) -> Path:
    """Create the output directory if it does not exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_raw_data(
    input_path: str,
) -> pl.DataFrame:
    """Load the raw wine dataset from a parquet file."""
    df = pl.read_parquet(input_path)
    logger.info(f"Loaded raw data with shape: {df.shape}")
    return df


def _create_derived_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create 4 derived features from raw wine measurements."""
    df = df.with_columns(
        [
            (pl.col("alcohol") / pl.col("malic_acid")).alias("alcohol_to_malic_ratio"),
            (pl.col("total_phenols") * pl.col("flavanoids")).alias("total_phenols_x_flavanoids"),
            pl.col("color_intensity").log1p().alias("color_intensity_log"),
            (pl.col("ash") * pl.col("alcalinity_of_ash")).alias("ash_alkalinity_interaction"),
        ]
    )
    logger.info(f"Created 4 derived features. New shape: {df.shape}")
    logger.info(f"Columns: {df.columns}")
    return df


def _handle_infinite_values(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Replace any infinite values in float columns with the column median."""
    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        if df[col].dtype in [pl.Float64, pl.Float32]:
            finite_vals = df[col].filter(df[col].is_finite())
            if finite_vals.len() > 0:
                median_val = float(finite_vals.median())
                df = df.with_columns(
                    pl.when(pl.col(col).is_infinite())
                    .then(median_val)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
    logger.info("Infinite value replacement complete.")
    return df


def _stratified_split(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Perform a stratified 80/20 train/test split preserving class proportions."""
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    x = df.select(feature_cols).to_numpy()
    y = df[TARGET_COLUMN].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = pl.DataFrame(
        {col: x_train[:, i] for i, col in enumerate(feature_cols)}
    ).with_columns(pl.Series(TARGET_COLUMN, y_train))

    test_df = pl.DataFrame({col: x_test[:, i] for i, col in enumerate(feature_cols)}).with_columns(
        pl.Series(TARGET_COLUMN, y_test)
    )

    logger.info(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")
    return train_df, test_df


def _scale_features(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, StandardScaler]:
    """Fit StandardScaler on train, transform both train and test sets."""
    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(train_df.select(feature_cols).to_numpy())
    x_test_scaled = scaler.transform(test_df.select(feature_cols).to_numpy())

    scaled_train = pl.DataFrame(
        {col: x_train_scaled[:, i] for i, col in enumerate(feature_cols)}
    ).with_columns(train_df[TARGET_COLUMN])

    scaled_test = pl.DataFrame(
        {col: x_test_scaled[:, i] for i, col in enumerate(feature_cols)}
    ).with_columns(test_df[TARGET_COLUMN])

    logger.info(f"Scaled {len(feature_cols)} features (scaler fit on train only).")
    return scaled_train, scaled_test, scaler


def _save_outputs(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str],
    output_path: Path,
) -> None:
    """Save train/test parquet files, fitted scaler, and feature name list."""
    train_df.write_parquet(output_path / "wine_train.parquet")
    test_df.write_parquet(output_path / "wine_test.parquet")
    joblib.dump(scaler, output_path / "scaler.joblib")
    (output_path / "feature_names.json").write_text(json.dumps(feature_cols, indent=2))
    logger.info(f"Saved train/test splits, scaler, and feature names to {output_path}")


def run_feature_engineering() -> None:
    """Run the full feature engineering pipeline."""
    start_time = time.time()
    logger.info("Starting feature engineering for UCI Wine dataset...")

    output_path = _ensure_output_dir(OUTPUT_DIR)
    df = _load_raw_data(INPUT_PATH)
    df = _create_derived_features(df)
    df = _handle_infinite_values(df)

    train_df, test_df = _stratified_split(df)
    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
    scaled_train, scaled_test, scaler = _scale_features(train_df, test_df)

    _save_outputs(scaled_train, scaled_test, scaler, feature_cols, output_path)

    elapsed = time.time() - start_time
    logger.info(f"Feature engineering completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    run_feature_engineering()
