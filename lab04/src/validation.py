
"""
Lab 4: Data Pipeline and Quality
Task 3: Schema, range, and missing-value validation.

Checks:
- Expected columns and data types
- Required numerical ranges
- Actual missing values and '?' indicators
- Target labels
- Validation violations without silently deleting rows
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "lab04" / "results"

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
    "income",
]

EXPECTED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

NUMERIC_RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}

EXPECTED_TARGET_LABELS = {"<=50K", ">50K"}


def load_dataset():
    """Load the original Adult dataset without cleaning it."""

    adult = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
    )

    X = adult.data.copy()
    y = adult.target.copy()

    # Keep the original data unchanged; combine copies for inspection.
    df = X.copy()
    df["income"] = y

    return df


def validate_dataset(df):
    """Run schema, type, range, missing-value, and target checks."""

    violations = []
    warnings = []

    print("=" * 65)
    print("LAB 04 - TASK 3: DATA VALIDATION")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Dataset dimensions
    # --------------------------------------------------

    print("\n1. Dataset dimensions")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if len(df) != 48842:
        violations.append(
            f"Expected 48842 rows, found {len(df)}."
        )

    if len(df.columns) != 15:
        violations.append(
            f"Expected 15 columns, found {len(df.columns)}."
        )

    # --------------------------------------------------
    # 2. Schema validation
    # --------------------------------------------------

    print("\n2. Schema validation")

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    unexpected_columns = [
        col for col in df.columns
        if col not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        violations.append(
            f"Missing required columns: {missing_columns}"
        )

    if unexpected_columns:
        violations.append(
            f"Unexpected columns: {unexpected_columns}"
        )

    if not missing_columns and not unexpected_columns:
        print("PASS: All expected columns are present.")

    # --------------------------------------------------
    # 3. Data type validation
    # --------------------------------------------------

    print("\n3. Data types")
    print(df.dtypes.to_string())

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue

        if not pd.api.types.is_integer_dtype(df[col].dtype):
            violations.append(
                f"{col}: expected integer type, found {df[col].dtype}."
            )

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue

        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            violations.append(
                f"{col}: expected category type, found {df[col].dtype}."
            )

    # --------------------------------------------------
    # 4. Missing-value checks
    # --------------------------------------------------

    print("\n4. Missing-value checks")

    actual_missing = df.isna().sum()

    question_mark_counts = pd.Series({
        col: int(
            df[col]
            .astype("object")
            .eq("?")
            .sum()
        )
        for col in df.columns
    })

    print("\nActual missing values (NaN):")
    if actual_missing.sum() == 0:
        print("None found.")
    else:
        print(actual_missing[actual_missing > 0].to_string())

    print("\nLiteral '?' indicators:")
    if question_mark_counts.sum() == 0:
        print("None found.")
    else:
        print(
            question_mark_counts[
                question_mark_counts > 0
            ].to_string()
        )

    for col in df.columns:
        missing_count = int(actual_missing[col])
        question_count = int(question_mark_counts[col])

        if missing_count or question_count:
            warnings.append(
                f"{col}: {missing_count} NaN values and "
                f"{question_count} literal '?' indicators."
            )

    print(
        "\nPolicy: '?' is treated as a missing-value indicator "
        "for cleaning."
    )
    print(
        "Validation does not modify the source data or remove rows."
    )

    # --------------------------------------------------
    # 5. Numerical range checks
    # --------------------------------------------------

    print("\n5. Numerical range checks")

    range_results = []

    for col, (minimum, maximum) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        invalid_mask = (
            values.notna()
            & ((values < minimum) | (values > maximum))
        )

        invalid_count = int(invalid_mask.sum())

        range_results.append({
            "column": col,
            "minimum": minimum,
            "maximum": maximum,
            "violations": invalid_count,
        })

        print(
            f"{col}: allowed range [{minimum}, {maximum}], "
            f"violations = {invalid_count}"
        )

        if invalid_count:
            violations.append(
                f"{col}: {invalid_count} values outside "
                f"[{minimum}, {maximum}]."
            )

    # --------------------------------------------------
    # 6. Target-label validation
    # --------------------------------------------------

    print("\n6. Target-label validation")

    target = df["income"].astype("string").str.strip()

    observed_labels = set(target.dropna().unique())
    unexpected_labels = observed_labels - EXPECTED_TARGET_LABELS

    missing_target_count = int(target.isna().sum())

    print(f"Observed labels: {sorted(observed_labels)}")
    print(f"Missing target labels: {missing_target_count}")

    if unexpected_labels:
        violations.append(
            f"Unexpected target labels: {sorted(unexpected_labels)}"
        )

    if missing_target_count:
        violations.append(
            f"Target has {missing_target_count} missing values."
        )

    # --------------------------------------------------
    # 7. Save the validation report
    # --------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIR / "validation_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("LAB 04 - DATA VALIDATION REPORT\n")
        report.write("=" * 45 + "\n\n")

        report.write("DATASET DIMENSIONS\n")
        report.write(f"Rows: {len(df)}\n")
        report.write(f"Columns: {len(df.columns)}\n\n")

        report.write("DATA TYPES\n")
        report.write(df.dtypes.to_string())
        report.write("\n\n")

        report.write("ACTUAL MISSING VALUES\n")
        report.write(actual_missing.to_string())
        report.write("\n\n")

        report.write("LITERAL QUESTION-MARK INDICATORS\n")
        report.write(question_mark_counts.to_string())
        report.write("\n\n")

        report.write("NUMERICAL RANGE CHECKS\n")
        for result in range_results:
            report.write(
                f"{result['column']}: "
                f"range [{result['minimum']}, "
                f"{result['maximum']}], "
                f"violations = {result['violations']}\n"
            )

        report.write("\nVALIDATION VIOLATIONS\n")
        if violations:
            for item in violations:
                report.write(f"- {item}\n")
        else:
            report.write("No schema, type, range, or target violations.\n")

        report.write("\nDATA-QUALITY WARNINGS\n")
        if warnings:
            for item in warnings:
                report.write(f"- {item}\n")
        else:
            report.write("No missing-value warnings.\n")

        report.write(
            "\nNo rows were removed or modified during validation.\n"
        )

    # --------------------------------------------------
    # 8. Summary
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Violations: {len(violations)}")
    print(f"Data-quality warnings: {len(warnings)}")

    for item in violations:
        print(f"VIOLATION: {item}")

    for item in warnings:
        print(f"WARNING: {item}")

    print(f"\nReport saved to: {report_path}")
    print("Validation completed. No rows were removed.")

    return violations, warnings


if __name__ == "__main__":
    dataset = load_dataset()
    validate_dataset(dataset)

