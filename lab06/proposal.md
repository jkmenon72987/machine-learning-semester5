# Individual Project Proposal

**Project Title:** Lightweight Fashion Image Classification for Edge Devices  
**Student Name:** Jishnu Krishna Menon  
**Student ID:** 72987  
**University:** Vistula University  
**Programme:** Bachelor of Engineering in Artificial Intelligence  

## 1. Problem Statement

Image classification is an important application of artificial intelligence and machine learning. However, deploying deep learning models on edge devices can be challenging because these devices have limited memory, processing power, and computational resources.

The aim of this project is to develop and optimize a lightweight Convolutional Neural Network (CNN) that can classify clothing images efficiently while maintaining good classification accuracy. The project will focus on reducing model size and inference time so that the model can be suitable for resource-constrained devices.

## 2. Dataset

The project will use the **Fashion-MNIST dataset**, which contains 70,000 grayscale images of clothing items across 10 different classes.

- Training images: 60,000
- Test images: 10,000
- Image resolution: 28 × 28 pixels
- Number of classes: 10

The dataset will be divided into training, validation, and testing sets for model development and evaluation.

## 3. Model and Development Approach

A Convolutional Neural Network (CNN) will be used as the baseline model. The model will be trained using PyTorch and evaluated using the validation and test datasets.

The project will investigate techniques to improve model efficiency while preserving classification performance.

## 4. Success Metrics

The model will be evaluated using the following criteria:

- **Classification accuracy:** Maintain high accuracy on the Fashion-MNIST test dataset.
- **Inference latency:** Target no more than 50 milliseconds per prediction.
- **Model size:** Target a model size of no more than 5 MB.
- **Efficiency:** Reduce computational requirements without significantly affecting classification accuracy.

## 5. Deployment Constraints

The final model should be suitable for deployment on devices with limited computational resources.

The main constraints are:

- Maximum inference latency: 50 ms.
- Maximum model size: 5 MB.
- Limited memory and processing capacity.
- Reliable classification performance.

## 6. Planned Optimization Techniques

The following optimization techniques will be investigated:

1. **Pruning:** Remove less important model weights to reduce the number of parameters.
2. **Quantization:** Reduce numerical precision to decrease model size and potentially improve inference speed.
3. **Knowledge Distillation:** Train a smaller student model using the knowledge of a larger teacher model.

The effectiveness of these techniques will be compared against the original CNN model.

## 7. Expected Outcome

The expected outcome is a compact and efficient image classification model that can classify Fashion-MNIST images while satisfying the specified model-size and latency targets.

The project will also provide a comparison of the baseline and optimized models in terms of accuracy, model size, and inference speed.

**Project Status:** Proposed – subject to instructor approval.