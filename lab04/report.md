# Lab 4 Report: Data Pipeline and Quality

## Executive summary

This lab builds a reproducible data-quality and modeling pipeline for the UCI Adult Income dataset, loaded from OpenML (version 2). The task is to predict whether annual income is greater than `$50K` using 14 demographic and employment-related input features. The pipeline validates the dataset, handles missing values and outliers, engineers features, evaluates Logistic Regression, and audits label consistency.

On the fixed stratified test split, adding `capital_net` and `hours_per_age` improved accuracy from **85.40% to 85.78%** and F1 from **0.6601 to 0.6696**. The gains are modest and should be interpreted as results from one held-out split, not as evidence of broad generalization.

## Objective and dataset

The goal is to create a reproducible, leakage-aware workflow and compare a baseline Logistic Regression model with one using two derived numeric features. The positive class is `>50K`; `<=50K` is the other class.

| Property | Value |
| --- | --- |
| Source | UCI Adult dataset via OpenML (`adult`, version 2) |
| Rows | 48,842 |
| Input features | 14 |
| Target | `income` |
| Classes | `<=50K`: 37,155; `>50K`: 11,687 |
| Random seed | 42 |
| Split | Stratified, 70% train / 30% test |
| Dataset SHA-256 | `88bb78216c6bac623cefa70797ad1177303c21f69558006dad76cbab6071116c` |

## Data validation and quality

Validation found no schema, type, range, or target violations. The dataset has 15 columns including the target. Missing values were found in three categorical input fields:

| Feature | Missing values |
| --- | ---: |
| `workclass` | 2,799 |
| `occupation` | 2,809 |
| `native-country` | 857 |

No literal question-mark indicators remained after ingestion, and no records were changed or removed during validation. The checked numeric ranges were age 17–90, hours per week 1–99, capital gain 0–99,999, and capital loss 0–4,356; all passed.

The label-quality audit identified **five groups** with identical recorded feature values but conflicting income labels, involving **10 records**. This indicates ambiguity worth reviewing, but does not by itself establish that any particular label is incorrect. A random sample of 50 records was also saved for manual review.

## Method

1. Load Adult from OpenML and normalize missing-value markers.
2. Validate columns, data types, target values, missingness, and selected numeric ranges.
3. Create a stratified train/test split using seed 42, then remove exact feature-and-label duplicates from the training set only. This removed 24 training rows: 34,189 before deduplication and 34,165 after; the test set contains 14,653 rows.
4. Fit preprocessing on training data only:
   - Numeric missing values: median imputation.
   - Categorical missing values: most-frequent-value imputation.
   - Numeric outliers: clipping at training-set 1st and 99th percentiles.
   - Numeric scaling: `StandardScaler`.
   - Categorical encoding: one-hot encoding, ignoring unseen categories.
5. Fit Logistic Regression (`max_iter=1000`) and evaluate accuracy and positive-class F1 on the same held-out test set.
6. Run feature and label-quality audits and save the generated reports and artifacts.

The preprocessing report records 105 transformed columns and confirms that test data was not used to fit transformations. Feature engineering in the model comparison is performed after splitting:

- `capital_net = capital-gain - capital-loss`
- `hours_per_age = hours-per-week / (age - 16)`

The feature audit also records an `age_group` feature, with four age bands. The model comparison CSV specifically compares the original features against the original features plus the two required numeric derived features; it does not report a separate model result for `age_group`.

## Model results

| Experiment | Accuracy | F1 (`>50K`) | Training rows | Test rows |
| --- | ---: | ---: | ---: | ---: |
| Baseline: original features | 0.8540 | 0.6601 | 34,165 | 14,653 |
| With `capital_net` and `hours_per_age` | 0.8578 | 0.6696 | 34,165 | 14,653 |

Adding the two engineered features increased accuracy by about **0.38 percentage points** and F1 by about **0.0095** on this split. This is a useful directional comparison, but the lab does not report repeated splits, cross-validation, confidence intervals, or significance testing. The result should therefore not be treated as proof that the engineered features will improve performance on other samples.

## Reproducibility and artifacts

The pipeline is configured with seed 42, a stratified 30% test set, and SHA-256 dataset hashing. To run all stages from the repository root:

```bash
python lab04/src/run_pipeline.py
```

Generated outputs are stored in `lab04/results/`, including ingestion, validation, and preprocessing reports; feature and label audit outputs; model comparison metrics; the saved preprocessor; and the dataset hash. The data card and `config.yaml` document the source, processing choices, and expected artifacts.

## Limitations and responsible use

The dataset reflects historical demographic and employment information and may not represent present-day populations or labor-market conditions. Demographic fields may encode historical social and economic inequalities, and a model can reproduce or amplify those patterns. Missing values and conflicting labels also add uncertainty. This educational model should not be used as the sole basis for employment, financial, or other consequential decisions. Verify the dataset's applicable license and usage conditions against its authoritative documentation before redistribution.

## Conclusion

The lab produces a complete, reproducible pipeline with training-only preprocessing, data-quality checks, feature engineering, model evaluation, and label auditing. The engineered-feature model modestly outperformed the baseline on the recorded test split. Further evaluation across multiple splits and a careful review of the conflicting-label groups would be needed before drawing stronger conclusions.
