import logging
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output")


def _load_raw_data(
    filepath: Path,
) -> pl.DataFrame:
    """Load raw parquet data from path."""
    logging.info(f"Loading raw data from {filepath}")
    return pl.read_parquet(filepath)


def _split_data(
    df: pl.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """Split the dataframe into train and test sets."""
    logging.info("Splitting data into training and test sets")

    # Extract features and target
    X = df.drop("target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_pandas(),
        y.to_pandas(),
        test_size=test_size,
        random_state=random_state,
        stratify=y.to_pandas(),
    )

    # Convert back to polars
    X_train_pl = pl.DataFrame(X_train)
    X_test_pl = pl.DataFrame(X_test)
    y_train_pl = pl.Series("target", y_train)
    y_test_pl = pl.Series("target", y_test)

    return X_train_pl, X_test_pl, y_train_pl, y_test_pl


def _scale_features(
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply StandardScaler to train and test features."""
    logging.info("Applying StandardScaler to features")
    scaler = StandardScaler()

    # Fit and transform
    X_train_scaled = scaler.fit_transform(X_train.to_pandas())
    X_test_scaled = scaler.transform(X_test.to_pandas())

    # Back to polars with original column names
    X_train_scaled_pl = pl.DataFrame(X_train_scaled, schema=X_train.columns)
    X_test_scaled_pl = pl.DataFrame(X_test_scaled, schema=X_test.columns)

    return X_train_scaled_pl, X_test_scaled_pl


def main() -> None:
    """Run feature engineering pipeline."""
    raw_path = OUTPUT_DIR / "wine_raw.parquet"
    if not raw_path.exists():
        logging.error(f"Raw data file not found at {raw_path}")
        return

    df = _load_raw_data(raw_path)
    X_train, X_test, y_train, y_test = _split_data(df)
    X_train_scaled, X_test_scaled = _scale_features(X_train, X_test)

    # Reattach targets and save
    train_df = X_train_scaled.with_columns(y_train)
    test_df = X_test_scaled.with_columns(y_test)

    train_path = OUTPUT_DIR / "wine_train.parquet"
    test_path = OUTPUT_DIR / "wine_test.parquet"

    train_df.write_parquet(train_path)
    test_df.write_parquet(test_path)

    logging.info(f"Saved engineered training set to {train_path}")
    logging.info(f"Saved engineered testing set to {test_path}")


if __name__ == "__main__":
    main()
