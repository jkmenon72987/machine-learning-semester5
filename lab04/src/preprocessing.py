
"""
Lab 4: Data Pipeline and Quality
Task 4: Leakage-aware data cleaning and transformation.

Operations:
1. Load the Adult dataset.
2. Split into training and testing sets before preprocessing.
3. Remove exact duplicate feature-and-label records from training only.
4. Impute missing numerical and categorical values.
5. Clip numerical values using training-set quantiles.
6. Standardize numerical features.
7. One-hot encode categorical features.
8. Save the fitted preprocessing pipeline and a report.

Important:
All learned preprocessing parameters are fitted on training data only.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from quantile_clipper import TrainingQuantileClipper


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "lab04" / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.30

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

EXPECTED_TARGET_LABELS = {"<=50K", ">50K"}


# --------------------------------------------------
# 2. Load the original dataset
# --------------------------------------------------

def load_dataset():
    """Load the Adult dataset without fitting transformations."""

    adult = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
    )

    X = adult.data.copy()
    y = adult.target.copy()

    # Treat literal question marks as missing values too.
    X = X.replace("?", np.nan)

    # Normalize target labels to strings for validation.
    y = y.astype("object").astype("string").str.strip()

    return X, y


# --------------------------------------------------
# 3. Validate the input schema
# --------------------------------------------------

def validate_schema(X, y):
    """Check that required features and labels are present."""

    expected_features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    missing_features = [
        col for col in expected_features if col not in X.columns
    ]

    unexpected_features = [
        col for col in X.columns if col not in expected_features
    ]

    if missing_features:
        raise ValueError(
            f"Required features are missing: {missing_features}"
        )

    if unexpected_features:
        raise ValueError(
            f"Unexpected input features found: {unexpected_features}"
        )

    if y.isna().any():
        raise ValueError(
            "The target contains missing values. "
            "Investigate them before training."
        )

    observed_labels = set(y.unique())
    unexpected_labels = observed_labels - EXPECTED_TARGET_LABELS

    if unexpected_labels:
        raise ValueError(
            f"Unexpected target labels: {unexpected_labels}"
        )


# --------------------------------------------------
# 4. Remove exact duplicate training records
# --------------------------------------------------

def remove_training_duplicates(X_train, y_train):
    """
    Remove exact duplicate records from the training set only.

    A record is considered duplicated when all feature values and
    its target label match an earlier training record.
    """

    combined = X_train.copy()
    combined["_target_"] = y_train.to_numpy()

    rows_before = len(combined)

    combined = combined.drop_duplicates(keep="first")

    duplicates_removed = rows_before - len(combined)

    y_clean = combined.pop("_target_")
    X_clean = combined[X_train.columns]

    # Keep indices aligned between features and target.
    y_clean = pd.Series(
        y_clean.to_numpy(),
        index=X_clean.index,
        name=y_train.name,
    )

    return X_clean, y_clean, duplicates_removed


# --------------------------------------------------
# 5. Build the preprocessing pipeline
# --------------------------------------------------

def build_preprocessor():
    """Create preprocessing steps for numerical and categorical data."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "outlier_clipper",
                TrainingQuantileClipper(
                    lower_quantile=1.0,
                    upper_quantile=99.0,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# --------------------------------------------------
# 6. Main preprocessing workflow
# --------------------------------------------------

def main():
    print("=" * 65)
    print("LAB 04 - TASK 4: CLEANING AND TRANSFORMATION")
    print("=" * 65)

    # Load and validate the original dataset.
    X, y = load_dataset()
    validate_schema(X, y)

    print(f"\nOriginal rows: {len(X)}")
    print(f"Original input features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts().to_string()}")

    # Split before fitting imputers, clipping thresholds, or scalers.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    training_rows_before_deduplication = len(X_train)
    testing_rows = len(X_test)

    print("\nStratified split completed.")
    print(
        f"Training rows before duplicate removal: "
        f"{training_rows_before_deduplication}"
    )
    print(f"Testing rows: {testing_rows}")

    # Remove duplicates from training data only.
    X_train, y_train, duplicates_removed = remove_training_duplicates(
        X_train,
        y_train,
    )

    print(f"Training duplicates removed: {duplicates_removed}")
    print(f"Training rows after duplicate removal: {len(X_train)}")

    # Build the pipeline and fit it ONLY on training features.
    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform the test data without fitting any new parameters.
    X_test_processed = preprocessor.transform(X_test)

    print("\nPreprocessing completed.")

    print(f"Processed training shape: {X_train_processed.shape}")
    print(f"Processed testing shape: {X_test_processed.shape}")

    # Verify that processed data contains no NaN or infinity values.
    train_values = (
        X_train_processed.data
        if hasattr(X_train_processed, "data")
        else np.asarray(X_train_processed)
    )

    test_values = (
        X_test_processed.data
        if hasattr(X_test_processed, "data")
        else np.asarray(X_test_processed)
    )

    train_finite = bool(np.isfinite(train_values).all())
    test_finite = bool(np.isfinite(test_values).all())

    print(f"Training processed values are finite: {train_finite}")
    print(f"Testing processed values are finite: {test_finite}")

    if not train_finite or not test_finite:
        raise ValueError(
            "Non-finite values remain after preprocessing."
        )

    # Save the fitted pipeline.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preprocessor_path = RESULTS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    # Save a detailed run report.
    report_path = RESULTS_DIR / "preprocessing_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("LAB 04 - PREPROCESSING REPORT\n")
        report.write("=" * 45 + "\n\n")

        report.write("CONFIGURATION\n")
        report.write(f"Random seed: {RANDOM_STATE}\n")
        report.write(f"Test fraction: {TEST_SIZE}\n")
        report.write("Split strategy: stratified\n\n")

        report.write("DATASET SIZES\n")
        report.write(f"Original rows: {len(X)}\n")
        report.write(
            f"Training rows before deduplication: "
            f"{training_rows_before_deduplication}\n"
        )
        report.write(f"Test rows: {testing_rows}\n")
        report.write(f"Training duplicates removed: {duplicates_removed}\n")
        report.write(f"Training rows after deduplication: {len(X_train)}\n\n")

        report.write("PROCESSED SHAPES\n")
        report.write(f"Training: {X_train_processed.shape}\n")
        report.write(f"Testing: {X_test_processed.shape}\n\n")

        report.write("DATA QUALITY CHECKS\n")
        report.write(f"Training values finite: {train_finite}\n")
        report.write(f"Test values finite: {test_finite}\n\n")

        report.write("PREPROCESSING OPERATIONS\n")
        report.write("- Numerical missing values: training median\n")
        report.write("- Categorical missing values: training mode\n")
        report.write("- Outlier clipping: training 1st and 99th percentiles\n")
        report.write("- Numerical scaling: StandardScaler\n")
        report.write("- Categorical encoding: OneHotEncoder\n")
        report.write("- Duplicate removal: training set only\n")
        report.write("- Test data used to fit transformations: No\n\n")

        report.write("SAVED ARTIFACT\n")
        report.write(f"Preprocessor: {preprocessor_path}\n")

    print(f"\nSaved preprocessor: {preprocessor_path}")
    print(f"Saved report: {report_path}")
    print("\nTask 4 preprocessing completed successfully.")

    # Return data for later model training and evaluation.
    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )


if __name__ == "__main__":
    main()

