
"""Lab 4, Task 6: Label-quality audit for the Adult dataset."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SAMPLE_SIZE = 50


def load_dataset():
    """Load the original Adult dataset without removing records."""
    dataset = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
    )

    X = dataset.data.copy()
    y = dataset.target.copy()

    # Keep a consistent target-column name.
    df = X.copy()
    df["income"] = y.astype(str).str.strip()

    # Handle missing-value markers if present.
    df = df.replace("?", pd.NA)

    return df, list(X.columns)


def inspect_random_sample(df):
    """Save 50 randomly selected rows for manual inspection."""
    sample_size = min(SAMPLE_SIZE, len(df))

    sample = df.sample(
        n=sample_size,
        random_state=RANDOM_SEED,
    )

    sample_path = RESULTS_DIR / "label_audit_sample_50.csv"
    sample.to_csv(sample_path, index=False)

    # Basic indicators to guide manual inspection.
    missing_by_column = sample.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0]

    return sample, sample_path, missing_by_column


def find_conflicting_labels(df, feature_columns):
    """
    Find groups with identical original features but different income labels.
    """
    grouped = (
        df.groupby(
            feature_columns,
            dropna=False,
            observed=True,
        )["income"]
        .agg(
            record_count="size",
            distinct_labels="nunique",
        )
        .reset_index()
    )

    conflicts = grouped[grouped["distinct_labels"] > 1].copy()

    if conflicts.empty:
        conflicting_records = df.iloc[0:0].copy()
    else:
        conflict_keys = conflicts[feature_columns]

        # Merge the conflicting feature groups back to the original data.
        conflicting_records = df.merge(
            conflict_keys,
            on=feature_columns,
            how="inner",
        )

    return conflicts, conflicting_records


def main():
    print("=" * 65)
    print("LAB 04 - TASK 6: LABEL-QUALITY AUDIT")
    print("=" * 65)

    df, feature_columns = load_dataset()

    print(f"\nDataset rows: {len(df)}")
    print(f"Original input features: {len(feature_columns)}")

    # Part A: Random sample for manual inspection.
    sample, sample_path, missing_by_column = inspect_random_sample(df)

    print(f"\nPart A: Random sample inspection")
    print(f"Sample size: {len(sample)}")
    print(f"Sample saved to: {sample_path}")

    if missing_by_column.empty:
        print("Missing values in sample: none")
    else:
        print("\nMissing values in the sampled rows:")
        print(missing_by_column.to_string())

    # Part B: Conflicting labels among identical feature rows.
    conflicts, conflicting_records = find_conflicting_labels(
        df,
        feature_columns,
    )

    conflict_groups_path = RESULTS_DIR / "conflicting_label_groups.csv"
    conflict_records_path = RESULTS_DIR / "conflicting_label_records.csv"

    conflicts.to_csv(conflict_groups_path, index=False)
    conflicting_records.to_csv(conflict_records_path, index=False)

    print("\nPart B: Conflicting target-label audit")
    print(f"Conflicting feature groups: {len(conflicts)}")
    print(f"Records in conflicting groups: {len(conflicting_records)}")

    if not conflicts.empty:
        print("\nIncome labels among conflicting records:")
        print(
            conflicting_records["income"]
            .value_counts(dropna=False)
            .to_string()
        )

    # Create a human-readable report.
    report_path = RESULTS_DIR / "label_quality_audit_report.txt"

    report_lines = [
        "LAB 04 - TASK 6: LABEL-QUALITY AUDIT",
        "=" * 50,
        f"Dataset rows: {len(df)}",
        f"Original input features: {len(feature_columns)}",
        "",
        "PART A: RANDOM SAMPLE",
        f"Sample size: {len(sample)}",
        f"Random seed: {RANDOM_SEED}",
        f"Sample CSV: {sample_path.name}",
        "",
        "Missing values in the sample:",
    ]

    if missing_by_column.empty:
        report_lines.append("None detected.")
    else:
        for column, count in missing_by_column.items():
            report_lines.append(f"{column}: {count}")

    report_lines.extend([
        "",
        "PART B: CONFLICTING LABELS",
        f"Conflicting feature groups: {len(conflicts)}",
        f"Records in conflicting groups: {len(conflicting_records)}",
        f"Conflict-group CSV: {conflict_groups_path.name}",
        f"Conflict-record CSV: {conflict_records_path.name}",
        "",
        "MANUAL REVIEW REQUIRED",
        (
            "Inspect label_audit_sample_50.csv against the dataset "
            "documentation. Record suspicious or inconsistent values "
            "only when supported by the data dictionary."
        ),
        (
            "Conflicting feature groups are records with identical "
            "original feature values but different income labels. "
            "A conflict does not automatically prove a label is wrong."
        ),
    ])

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {report_path}")
    print("\nTask 6 audit completed.")


if __name__ == "__main__":
    main()

