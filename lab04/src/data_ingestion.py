
"""
Lab 4: Data Pipeline and Quality
Task 2: Data ingestion.

Load the original Adult dataset without cleaning or modifying it.
"""

from pathlib import Path
from sklearn.datasets import fetch_openml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "lab04" / "results"


def load_raw_data():
    """Load and inspect the original Adult dataset."""

    print("=" * 60)
    print("LAB 04 - DATA INGESTION")
    print("=" * 60)

    print("\nLoading the Adult dataset...")

    adult = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
    )

    # Keep copies of the original input features and target.
    X = adult.data.copy()
    y = adult.target.copy()

    # Combine copies for inspection; do not clean the data here.
    df = X.copy()
    df["income"] = y

    print("\nDataset loaded successfully.")
    print("Source: OpenML Adult, version 2")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Input features: {X.shape[1]}")
    print("Target: income")

    print("\nColumn names:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst five rows:")
    print(df.head())

    print("\nQuestion-mark indicators by column:")
    for column in df.columns:
        values = df[column].astype("object")
        count = int((values == "?").sum())

        if count > 0:
            print(f"{column}: {count}")

    # Save the ingestion report.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "ingestion_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("LAB 04 - DATA INGESTION REPORT\n")
        report.write("=" * 40 + "\n")
        report.write("Dataset: OpenML Adult, version 2\n")
        report.write(f"Rows: {df.shape[0]}\n")
        report.write(f"Columns: {df.shape[1]}\n")
        report.write(f"Input features: {X.shape[1]}\n")
        report.write("Target column: income\n\n")
        report.write("Columns:\n")

        for column in df.columns:
            report.write(f"- {column}\n")

        report.write("\nData types:\n")
        report.write(df.dtypes.to_string())
        report.write("\n")

    print(f"\nReport saved to: {report_path}")
    print("\nNo cleaning or row removal was performed.")

    return X, y, df


if __name__ == "__main__":
    load_raw_data()

