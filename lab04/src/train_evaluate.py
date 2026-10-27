
"""
Lab 4: Data Pipeline and Quality
Task 5: Required feature engineering and Logistic Regression comparison.

Experiments:
1. Baseline using the original 14 features.
2. Model using the original features plus capital_net and hours_per_age.

Both experiments use the same stratified train/test split.
Preprocessing is fitted on training data only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from quantile_clipper import TrainingQuantileClipper


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

DERIVED_NUMERIC_COLUMNS = [
    "capital_net",
    "hours_per_age",
]


def load_dataset():
    """Load the Adult dataset."""
    adult = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
    )

    X = adult.data.copy()
    y = adult.target.astype(str).str.strip().copy()

    X = X.replace("?", np.nan)

    return X, y


def add_required_features(X):
    """Create the two features required by the lab manual."""
    result = X.copy()

    result["capital_net"] = (
        pd.to_numeric(result["capital-gain"], errors="coerce")
        - pd.to_numeric(result["capital-loss"], errors="coerce")
    )

    age = pd.to_numeric(result["age"], errors="coerce")
    hours = pd.to_numeric(result["hours-per-week"], errors="coerce")

    # The expected age range is 17-90, so age - 16 is positive.
    result["hours_per_age"] = hours / (age - 16)

    return result


def remove_training_duplicates(X_train, y_train):
    """Remove exact feature-and-label duplicates from training only."""
    combined = X_train.copy()
    combined["_target_"] = y_train.to_numpy()

    before = len(combined)
    combined = combined.drop_duplicates(keep="first")
    removed = before - len(combined)

    y_clean = combined.pop("_target_")
    X_clean = combined[X_train.columns]

    y_clean = pd.Series(
        y_clean.to_numpy(),
        index=X_clean.index,
        name=y_train.name,
    )

    return X_clean, y_clean, removed


def build_model(numeric_columns, categorical_columns):
    """Build preprocessing and Logistic Regression as one pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clipper",
                TrainingQuantileClipper(
                    lower_quantile=1,
                    upper_quantile=99,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    """Fit on training data and evaluate on the held-out test data."""
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(
        y_test,
        predictions,
        pos_label=">50K",
    )

    print(f"\n{name}")
    print("-" * 45)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")

    return {
        "experiment": name,
        "accuracy": accuracy,
        "f1_score": f1,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
    }


def main():
    print("=" * 65)
    print("LAB 04 - TASK 5: FEATURE ENGINEERING AND MODEL COMPARISON")
    print("=" * 65)

    X, y = load_dataset()

    # Split original records before fitting preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Remove exact duplicates from training only.
    X_train, y_train, duplicates_removed = remove_training_duplicates(
        X_train,
        y_train,
    )

    print(f"\nTraining duplicates removed: {duplicates_removed}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    # Experiment 1: original features.
    baseline_model = build_model(
        NUMERIC_COLUMNS,
        CATEGORICAL_COLUMNS,
    )

    baseline_result = evaluate_model(
        baseline_model,
        X_train,
        y_train,
        X_test,
        y_test,
        "Baseline: original features",
    )

    # Experiment 2: engineer features after the split.
    X_train_engineered = add_required_features(X_train)
    X_test_engineered = add_required_features(X_test)

    engineered_model = build_model(
        NUMERIC_COLUMNS + DERIVED_NUMERIC_COLUMNS,
        CATEGORICAL_COLUMNS,
    )

    engineered_result = evaluate_model(
        engineered_model,
        X_train_engineered,
        y_train,
        X_test_engineered,
        y_test,
        "With capital_net and hours_per_age",
    )

    results = pd.DataFrame([
        baseline_result,
        engineered_result,
    ])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "model_comparison.csv"
    results.to_csv(output_path, index=False)

    print("\nMODEL COMPARISON")
    print("=" * 65)
    print(results.to_string(index=False))
    print(f"\nResults saved to: {output_path}")

    print("\nTask 5 experiment completed.")


if __name__ == "__main__":
    main()

