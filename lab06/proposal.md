# Individual Project Proposal

## Project Title
**Lightweight SMS Spam Detection for Mobile Devices**

**Student Name:** Jishnu Krishna Menon  
**Student ID:** 72987

### 1. Problem Definition

SMS spam messages are commonly used for unwanted advertising, fraudulent offers and phishing attempts. Identifying these messages automatically can help users avoid suspicious communications.

The objective of this project is to develop a lightweight machine learning model that classifies SMS messages into two categories: spam and legitimate (ham).

The main focus is to create a compact and efficient model that can perform predictions locally on devices with limited computing resources.

### 2. Dataset

The project will use the publicly available **UCI SMS Spam Collection** dataset.

The dataset contains labelled SMS messages classified as spam or ham. It will be divided into training, validation and testing sets using stratified sampling to preserve the class distribution.

Text preprocessing and TF-IDF vectorization will be used to convert messages into numerical features suitable for model training.

### 3. Proposed Model and Methodology

A lightweight feed-forward neural network will be developed using PyTorch.

The proposed workflow includes:
- Cleaning and preprocessing SMS text.
- Converting messages into numerical features using TF-IDF.
- Training a small neural network for binary classification.
- Evaluating the original model.
- Applying dynamic INT8 quantization to reduce model size and potentially improve inference efficiency.

The original and optimized models will be compared under the same evaluation conditions.

### 4. Success Metrics

The project will evaluate the model using:
- Accuracy
- Precision
- Recall
- F1-score, particularly for the spam class
- Model size in MB
- Inference latency in milliseconds

The aim is to maintain strong spam-detection performance while reducing the model's storage and computational requirements.

### 5. Deployment Constraints

The intended deployment environment is a resource-constrained device running local inference.

The project will use the following targets:
- Model size: no more than 5 MB.
- Inference latency: no more than 50 milliseconds per message.

The measurements will be performed on the available laptop CPU.

### 6. Planned Optimization Technique

**Dynamic INT8 Quantization**

The project will investigate dynamic quantization of the neural network's linear layers. This technique reduces the numerical precision used for model weights and can reduce storage requirements and inference time.

The optimized model will be compared with the original FP32 model to measure changes in accuracy, model size and inference latency.

### 7. Expected Outcome

The expected outcome is a lightweight SMS spam classification model capable of distinguishing unwanted messages from legitimate messages.

The project will provide an experimental comparison between the original and quantized models, demonstrating the trade-off between predictive performance, model size and inference efficiency.

**Project Status:** Proposed – subject to instructor approval.
