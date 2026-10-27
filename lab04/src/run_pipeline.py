
"""Run the Lab 4 data-quality and modelling stages in sequence."""

from pathlib import Path
import subprocess
import sys

SRC_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "data_ingestion.py",
    "validation.py",
    "preprocessing.py",
    "feature_audit.py",
    "train_evaluate.py",
    "label_quality_audit.py",
]

def main():
    print("=" * 60)
    print("LAB 04 - COMPLETE PIPELINE")
    print("=" * 60)

    for script in SCRIPTS:
        script_path = SRC_DIR / script

        if not script_path.exists():
            raise FileNotFoundError(f"Required script not found: {script_path}")

        print(f"\nRunning {script} ...")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=SRC_DIR.parent.parent,
            check=False,
        )

        if result.returncode != 0:
            print(f"\nPipeline stopped: {script} failed.")
            sys.exit(result.returncode)

        print(f"Completed: {script}")

    print("\nAll pipeline stages completed successfully.")

if __name__ == "__main__":
    main()

