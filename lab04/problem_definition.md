# Lab 4: Data Pipeline and Quality

## 1. Problem Definition

### Objective

The objective of this laboratory is to develop a reproducible and leakage-free data pipeline using the UCI Adult dataset, obtained through OpenML. The task is to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related information. The experiment focuses on data ingestion, validation, cleaning, transformation, feature engineering, and data-quality assessment.

### Input Features

The dataset contains 14 input features describing individuals. These include age, workclass, final weight (`fnlwgt`), education, education-num, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country.

The dataset contains both numerical and categorical features. Therefore, the pipeline must support missing-value handling, numerical outlier clipping, numerical scaling, and categorical encoding.

### Output Target

The target is income class. The prediction task is to determine whether an individual's income exceeds $50,000 per year. The positive class is `>50K`, while `<=50K` represents income that does not exceed $50,000.

### Success Metrics

The baseline classification model will be Logistic Regression. Its performance will be evaluated using accuracy and F1 score. Accuracy measures the proportion of correct predictions, while F1 score balances precision and recall for the positive class.

Both metrics will be recorded for experiments with and without the required derived features: `capital_net` and `hours_per_age`. Comparing these results will help evaluate whether the engineered features improve predictive performance.

### Constraints and Reproducibility

The pipeline must avoid data leakage by splitting the dataset into training and testing sets before fitting preprocessing transformations. Missing-value imputers, outlier-clipping thresholds, and scaling parameters must be learned using training data only. The test set must remain separate from the fitting process.

A fixed random seed and stratified train/test splitting will support reproducibility. The pipeline will record data-quality checks, preserve evidence of preprocessing decisions, calculate a SHA256 hash of the processed dataset, and document the dataset in a data card.

The final implementation must be runnable through a single command and produce the reports and evaluation results required by the laboratory instructions.
