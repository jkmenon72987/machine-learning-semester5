# Individual Project Proposal

## Project Title
**Development of a Lightweight Heart Failure Outcome Prediction Model Using Clinical Parameters**

**Student Name:** Jishnu Krishna Menon  
**Student ID:** 72987

### 1. Problem Definition

Heart failure is a serious cardiovascular condition that requires continuous monitoring and appropriate clinical management. Identifying patterns associated with adverse patient outcomes is an important area of medical research.

This project aims to develop a lightweight machine learning model that predicts the occurrence of a recorded mortality event among heart failure patients using routinely collected clinical parameters.

The primary research objective is to investigate whether a compact neural network can provide useful predictive performance while satisfying strict model-size and inference-latency constraints.

### 2. Dataset

The project will use the publicly available UCI Heart Failure Clinical Records dataset.

The dataset contains 299 patient records and 12 clinical features, including age, ejection fraction, serum creatinine, serum sodium, diabetes, anaemia and other patient characteristics.

The target variable is `DEATH_EVENT`, representing whether a mortality event was recorded during the follow-up period.

Data preprocessing, feature scaling and stratified sampling will be performed before model training.

### 3. Proposed Model and Methodology

A small feed-forward neural network will be implemented using PyTorch.

The proposed workflow includes:
- Clinical dataset exploration and preprocessing.
- Feature scaling and stratified data splitting.
- Development and training of a compact neural network.
- Evaluation using stratified cross-validation.
- Application of INT8 dynamic quantization.
- Comparison of the original and optimized models.

### 4. Success Metrics

Model performance will be evaluated using:
- Recall (sensitivity).
- Precision.
- F1-score.
- ROC-AUC.
- Balanced accuracy.

Deployment performance will be assessed through:
- Model size in MB.
- CPU inference latency in milliseconds.

The objective is to maintain useful predictive performance while satisfying the specified deployment constraints.

### 5. Deployment Constraints

The intended deployment environment is a resource-constrained CPU-based system.

The project will target:
- Model size: no more than 5 MB.
- Inference latency: no more than 50 milliseconds per prediction.

All performance measurements will be conducted on the available laptop CPU.

### 6. Planned Optimization

**Dynamic INT8 Quantization**

Dynamic quantization will be investigated as an optimization technique for the neural network's linear layers.

The optimized model will be compared with its original FP32 version to evaluate changes in model size, inference latency and predictive performance.

### 7. Expected Outcome

The expected outcome is a compact clinical outcome prediction model that demonstrates the practical trade-offs between predictive performance, model size and inference efficiency.

The project will provide an experimental evaluation of model optimization techniques using a publicly available medical dataset.

**Project Status:** Proposed – subject to instructor approval.
