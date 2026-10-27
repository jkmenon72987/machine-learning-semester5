# Data Card: Adult Income Dataset

## 1. Dataset Overview

The Adult Income dataset is used to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related features.

## 2. Data Source

* Dataset: Adult
* Source: UCI Adult dataset, accessed through OpenML
* OpenML dataset name: `adult`
* Version requested: 2
* Loading method: `fetch_openml(name="adult", version=2, as_frame=True)`

## 3. Dataset Size

* Total records: 48,842
* Input features: 14
* Target: Income category
* Target classes: `<=50K` and `>50K`

## 4. Intended Use

This dataset is used for educational machine-learning exercises involving data ingestion, validation, cleaning, feature engineering, classification, and model evaluation.

## 5. Data Processing

The lab pipeline includes:

* Schema and range validation.
* Identification and handling of missing values.
* Stratified 70/30 training/test split.
* Numerical median imputation.
* Categorical most-frequent-value imputation.
* Numerical outlier clipping using training-set percentiles.
* Numerical scaling and categorical one-hot encoding.
* Feature engineering using `capital_net` and `hours_per_age`.

Preprocessing parameters must be fitted on training data only to prevent data leakage.

## 6. Data Quality

Missing values occur in categorical features, including `workclass`, `occupation`, and `native-country`.

The label-quality audit identified five groups with identical recorded feature values but conflicting income labels. These conflicts should be investigated rather than automatically treated as confirmed errors.

## 7. Potential Biases and Limitations

The dataset reflects historical demographic and employment information and may not represent present-day populations or labour-market conditions.

Demographic features may reflect historical social and economic inequalities. Models trained on this data may reproduce or amplify these patterns.

The model should not be used as the sole basis for real-world employment or financial decisions.

## 8. Privacy and Ethical Considerations

The dataset contains demographic and employment-related attributes. It should be handled responsibly, and model predictions should not be treated as definitive judgments about individuals.

## 9. Reproducibility

* Train/test split: Stratified 70/30.
* Random seed: 42.
* Hash algorithm: SHA256.
* Saved preprocessor: `results/preprocessor.joblib`.
* Reports and experiment results: `results/`.

## 10. Licensing

The dataset's applicable licence and usage conditions must be verified against the authoritative dataset documentation before redistribution or submission.
