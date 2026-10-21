# Lab 3: MLP, CNN, and Transformer Architectures for Fashion-MNIST

## 1. Goal

The goal of this laboratory was to implement and compare three neural network architectures: a Multi-Layer Perceptron (MLP), a Convolutional Neural Network (CNN), and an attention-based Transformer model. The models were evaluated using the Fashion-MNIST dataset to investigate their classification accuracy, computational cost, number of trainable parameters, and inference latency.

## 2. Method

### 2.1 Dataset Preparation

The Fashion-MNIST dataset was loaded using `torchvision.datasets.FashionMNIST`.

The official training dataset of 60,000 images was divided into:

* **Training set:** 55,000 images
* **Validation set:** 5,000 images
* **Test set:** 10,000 images

The official test set was kept separate from the training and validation sets.

Images were converted into tensors and normalized using a mean of 0.2860 and a standard deviation of 0.3530. A random seed of 42 was used to make the dataset split and experiments more reproducible.

The batch size was 128.

### 2.2 Model Architectures

Three neural network architectures were implemented using PyTorch.

**MLP (Multi-Layer Perceptron)**

The MLP uses fully connected layers. The input image is flattened into a vector of 784 features, followed by hidden layers containing 256 and 128 neurons. A final linear layer produces scores for the ten Fashion-MNIST classes.

**CNN (Convolutional Neural Network)**

The CNN contains two convolutional layers with 16 and 32 output channels, respectively. ReLU activation functions and max-pooling operations are used to extract spatial features and reduce the feature-map dimensions. The final classifier produces scores for the ten classes.

**Attention (Transformer Model)**

The attention-based model divides each 28 × 28 image into sixteen non-overlapping patches of size 7 × 7. Each patch contains 49 pixel values and is projected into a 64-dimensional embedding. Two Transformer encoder layers process the patch embeddings using four attention heads and a feed-forward dimension of 128. The patch representations are averaged and passed to a final classifier with ten outputs.

### 2.3 Training Configuration

All three models were trained using the following configuration:

* **Optimizer:** Adam
* **Learning rate:** 0.001
* **Number of epochs:** 5
* **Batch size:** 128
* **Loss function:** Cross-Entropy Loss
* **Random seed:** 42

The models were evaluated on the official Fashion-MNIST test set after training.

### 2.4 Computational Cost and Latency Measurements

The number of trainable parameters was calculated by summing the parameters that require gradients.

Hand-calculated FLOPs were estimated from the principal arithmetic operations in each architecture. One multiply-accumulate operation (MAC) was counted as approximately two floating-point operations (FLOPs).

THOP was used to profile the models and estimate their MAC counts. The corresponding THOP FLOPs estimates were calculated as twice the reported MAC counts.

Inference latency was measured using `time.perf_counter()` under `torch.no_grad()`. Ten warm-up iterations were performed before timing. The median latency was calculated over 100 runs for batch size 1 and 20 runs for batch size 128.

The measurements were performed on the available CPU environment.

## 3. Results

### 3.1 Model Performance

The test accuracy results after five training epochs are shown below.

| Model     | Test Accuracy |
| --------- | ------------: |
| MLP       |        87.50% |
| CNN       |        89.82% |
| Attention |        83.71% |

The CNN achieved the highest test accuracy, followed by the MLP and the attention-based model.

### 3.2 Trainable Parameters and FLOPs

| Model     | Trainable Parameters | Hand-Calculated FLOPs (Batch 1) | THOP MACs (Batch 1) | THOP Estimated FLOPs (Batch 1) |
| --------- | -------------------: | ------------------------------: | ------------------: | -----------------------------: |
| MLP       |              235,146 |                         469,504 |             234,752 |                        469,504 |
| CNN       |               20,490 |                       2,063,488 |           1,031,744 |                      2,063,488 |
| Attention |               70,794 |                       2,329,856 |              50,816 |                        101,632 |

The MLP and CNN show agreement between the hand-calculated FLOPs and the THOP estimates under the counting convention used in this experiment.

A substantial difference appears for the attention-based model. Its hand-calculated estimate includes the patch projection, attention-related matrix operations, query/key/value and output projections, feed-forward layers, and final classifier. However, the THOP estimate appears to count the patch projection and classifier without fully accounting for the internal Transformer operations.

Consequently, the THOP FLOPs value for the attention model should not be interpreted as its complete computational cost. The hand-calculated estimate provides a broader estimate of the principal arithmetic operations in this architecture.

The hand calculations do not include every operation. For example, some bias additions, activation functions, normalization operations, and other implementation details are excluded. Therefore, these FLOPs values are estimates rather than exhaustive counts of every floating-point operation.

### 3.3 Inference Latency

The following results were obtained from the latest computational cost measurement run.

| Model     | Batch 1 Median Latency (ms) | Batch 128 Median Latency (ms) |
| --------- | --------------------------: | ----------------------------: |
| MLP       |                      0.0696 |                        0.7799 |
| CNN       |                      0.2180 |                       13.1020 |
| Attention |                      0.5887 |                        7.7908 |

The MLP had the lowest measured inference latency for both batch sizes in this run. The CNN had higher latency than the MLP, while the attention-based model had the highest single-image latency.

For batch size 128, the CNN showed the highest measured latency. This illustrates that latency depends on the operations performed by a model, batch size, implementation, and hardware—not just the number of parameters.

Latency values can vary between runs, so the table represents measurements from one experimental run rather than universal performance guarantees.

### 3.4 Training Results

The following table summarizes the training time and test accuracy recorded during the five-epoch experiments.

| Model     | Training Time (seconds) | Test Accuracy |
| --------- | ----------------------: | ------------: |
| MLP       |                 93.6626 |        87.50% |
| CNN       |                137.8883 |        89.82% |
| Attention |                145.4126 |        83.71% |

In this experiment, the MLP completed training in the shortest time. The attention-based model took the longest, while the CNN achieved the highest test accuracy.

Training times depend on the execution environment and should be interpreted in the context of this experiment.

### 3.5 Trade-Off Plots

Two plots were generated from the experimental results:

* `results/accuracy_vs_parameters.png`
* `results/accuracy_vs_latency.png`

The accuracy-versus-parameters plot compares test accuracy with the number of trainable parameters.

The accuracy-versus-latency plot compares test accuracy with single-image inference latency.

Together, these plots help illustrate the trade-offs between model size, predictive performance, and inference speed.

The CNN provides the highest test accuracy with the fewest trainable parameters among the three models. The MLP offers the lowest single-image inference latency, while the attention-based model has higher single-image latency and lower test accuracy in this experiment.

## 4. Discussion and Conclusions

### 4.1 Accuracy

The CNN achieved the highest test accuracy at 89.82%, followed by the MLP at 87.50% and the attention-based model at 83.71%.

Under the selected architecture and training configuration, the CNN performed best on the Fashion-MNIST test set.

### 4.2 Model Size and Computational Cost

The CNN had the fewest trainable parameters, with 20,490, compared with 70,794 for the attention model and 235,146 for the MLP.

However, the CNN's hand-calculated FLOPs estimate was higher than the MLP's. This demonstrates that parameter count and computational cost measure different aspects of a neural network.

The attention model also illustrates why profiler results must be interpreted carefully. The THOP estimate was substantially lower than the hand-calculated estimate because the profiler did not appear to count all the internal Transformer operations.

### 4.3 Inference Speed

The MLP achieved the lowest measured latency for both batch sizes in this run. The CNN achieved the highest accuracy but had greater measured latency than the MLP. The attention-based model had the highest single-image latency.

The best architecture therefore depends on the intended application. A model selected for high accuracy may not be the fastest, and a model with fewer parameters is not necessarily the fastest for every workload.

### 4.4 Overall Conclusion

This laboratory demonstrated how MLP, CNN, and attention-based architectures differ in classification accuracy, parameter count, estimated FLOPs, training time, and inference latency.

The CNN offered the best test accuracy and the smallest parameter count in this experiment. The MLP offered the lowest measured inference latency, while the attention-based model did not achieve competitive accuracy under the selected five-epoch training configuration.

These findings highlight the importance of evaluating neural network architectures using multiple metrics rather than relying on accuracy or parameter count alone.

## 5. Limitations

Several limitations should be considered when interpreting the results:

1. **Limited training duration:** Each model was trained for only five epochs. Longer training or different hyperparameters could change the accuracy rankings.
2. **Hardware dependence:** Training time and inference latency depend on the CPU, software environment, and background processes.
3. **Latency variability:** The reported latency values are medians from a limited number of timed runs and may vary between executions.
4. **FLOPs estimation:** Hand-calculated FLOPs cover the principal arithmetic operations rather than every operation performed by the models.
5. **Profiler limitations:** THOP did not appear to account for all internal Transformer operations in the attention-based model. Its FLOPs estimate should therefore be interpreted with caution.
6. **Architecture-specific results:** The comparison reflects the exact model implementations and training configuration used in this laboratory. Different architectures, hyperparameters, or hardware may produce different results.

Despite these limitations, the experiment provides a useful comparison of model accuracy, size, computational cost, and inference latency on Fashion-MNIST.
