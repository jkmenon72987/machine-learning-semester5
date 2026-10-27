
"""
Lab 4: Data Pipeline and Quality
Task 5: Feature engineering and label-quality audit.
"""

from pathlib import Path
import hashlib
import json

import pandas as pd
from sklearn.datasets import fetch_openml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "lab04" / "results"


def load_dataset():
    """Load the original Adult dataset."""
    adult = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
    )

    X = adult.data.copy()
    y = adult.target.copy()

    return X, y


def calculate_sha256(df):
    """Calculate a reproducible hash of a canonical dataset representation."""
    canonical = df.to_csv(
        index=False,
        na_rep="<MISSING>",
        lineterminator="\n",
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def audit_labels(y):
    """Check the target labels and class distribution."""
    expected_labels = {"<=50K", ">50K"}

    normalized = y.astype("string").str.strip()
    observed_labels = set(normalized.dropna().unique())

    missing_labels = int(normalized.isna().sum())
    unexpected_labels = observed_labels - expected_labels

    print("\nLABEL-QUALITY AUDIT")
    print("-" * 45)
    print("Class distribution:")
    print(normalized.value_counts(dropna=False).to_string())
    print(f"\nMissing target labels: {missing_labels}")
    print(f"Unexpected target labels: {sorted(unexpected_labels)}")

    return {
        "missing_target_labels": missing_labels,
        "unexpected_target_labels": sorted(unexpected_labels),
        "class_distribution": {
            str(label): int(count)
            for label, count in normalized.value_counts(
                dropna=False
            ).items()
        },
    }


def engineer_features(X):
    """Create a derived feature without using the target label."""
    result = X.copy()

    # Convert age into a broad age group.
    result["age_group"] = pd.cut(
        result["age"],
        bins=[-float("inf"), 25, 45, 65, float("inf")],
        labels=[
            "25_or_younger",
            "26_to_45",
            "46_to_65",
            "over_65",
        ],
    )

    return result


def main():
    print("=" * 60)
    print("LAB 04 - FEATURE ENGINEERING AND LABEL AUDIT")
    print("=" * 60)

    X, y = load_dataset()

    # Hash the original feature data and target separately.
    original_data = X.copy()
    original_data["income"] = y.astype("string")

    dataset_hash = calculate_sha256(original_data)

    print("\nDATASET FINGERPRINT")
    print("-" * 45)
    print(f"Dataset rows: {len(original_data)}")
    print(f"Dataset columns: {len(original_data.columns)}")
    print(f"SHA256: {dataset_hash}")

    # Audit the target labels.
    label_report = audit_labels(y)

    # Create and inspect the derived feature.
    X_engineered = engineer_features(X)

    print("\nFEATURE ENGINEERING")
    print("-" * 45)
    print("New feature: age_group")
    print("Age-group counts:")
    print(X_engineered["age_group"].value_counts(
        dropna=False
    ).to_string())

    print(f"\nOriginal feature count: {X.shape[1]}")
    print(f"Feature count after engineering: {X_engineered.shape[1]}")

    # Save a machine-readable report.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": "Adult",
        "source": "OpenML",
        "version": 2,
        "rows": int(len(original_data)),
        "columns_including_target": int(len(original_data.columns)),
        "sha256": dataset_hash,
        "label_audit": label_report,
        "feature_engineering": {
            "new_feature": "age_group",
            "description": (
                "Age grouped into four ranges: <=25, 26-45, "
                "46-65, and >65."
            ),
            "feature_count_before": int(X.shape[1]),
            "feature_count_after": int(X_engineered.shape[1]),
        },
    }

    report_path = RESULTS_DIR / "feature_audit_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    # Save the derived features separately, without the target.
    features_path = RESULTS_DIR / "engineered_features.csv"
    X_engineered.to_csv(features_path, index=False)

    print(f"\nReport saved to: {report_path}")
    print(f"Engineered features saved to: {features_path}")
    print("\nFeature engineering and label audit completed.")


if __name__ == "__main__":
    main()

