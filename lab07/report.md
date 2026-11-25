# Lab 7: Compression – Pruning and Distillation

**Student:** Jishnu Krishna Menon  
**Student ID:** 72987  
**Subject:** Machine Learning

---

# 1. Introduction

The objective of this laboratory was to investigate neural network compression techniques, particularly magnitude pruning, structured pruning and knowledge distillation.

The experiments used the Fashion-MNIST dataset and PyTorch. Model performance was evaluated using classification accuracy, parameter count, storage size and CPU inference latency.

The laboratory also investigated the trade-offs between model compression, computational efficiency and classification performance.

# 2. Experimental Environment

| Component | Configuration |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch 2.3.1 |
| Dataset | Fashion-MNIST |
| Hardware | CPU |
| Training batch size | 128 |
| Random seed | 42 |
| Optimizer | Adam |
| Baseline learning rate | 0.001 |
| Operating environment | Windows / VS Code |

Dataset distribution:

| Dataset | Samples |
|---|---:|
| Training | 55,000 |
| Validation | 5,000 |
| Testing | 10,000 |

# 3. Task 1 – Oversized Baseline CNN

An oversized convolutional neural network was developed and trained using Fashion-MNIST.

The model contained three convolutional layers followed by fully connected classification layers.

Training configuration:

- Epochs: 5
- Batch size: 128
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss

## Baseline Results

| Metric | Result |
|---|---:|
| Parameters | 1,015,114 |
| Training time | 329.98 seconds |
| Test loss | 0.2584 |
| Test accuracy | 90.60% |
| Model size | 3.8763 MB |
| CPU inference latency | 1.2796 ms/image |

The baseline model achieved 90.60% test accuracy and served as the teacher model and reference model for subsequent experiments.

# 4. Task 2 – Magnitude and Structured Pruning

## 4.1 Global Magnitude Pruning

Global magnitude pruning ranks weights across the selected network layers according to their absolute values and removes the smallest weights.

The experiment used sparsity levels of 0%, 25%, 50%, 75%, 90% and 95%.

| Sparsity | Test accuracy |
|---:|---:|
| 0% | 90.60% |
| 25% | 92.02% |
| 50% | 91.63% |
| 75% | 91.27% |
| 90% | 91.44% |
| 95% | 91.03% |

Global pruning maintained good classification accuracy even at 95% sparsity.

## 4.2 Layer-wise Magnitude Pruning

Layer-wise pruning removes weights independently within each selected layer.

| Sparsity | Test accuracy |
|---:|---:|
| 0% | 90.60% |
| 25% | 91.87% |
| 50% | 91.85% |
| 75% | 91.15% |
| 90% | 88.13% |
| 95% | 82.88% |

Layer-wise pruning performed well at lower sparsity levels but experienced a significant accuracy reduction at 90% and 95%.

This suggests that applying an equal pruning ratio to every layer can damage layers that are more sensitive to weight removal.

## 4.3 Structured L2 Filter Pruning

Structured pruning was performed by calculating the L2 norms of convolutional filters and applying pruning masks.

The selected pruning levels were 10%, 25% and 50%.

| Target filter pruning | Actual filter sparsity | Test accuracy |
|---:|---:|---:|
| 10% | 9.82% | 91.94% |
| 25% | 25.00% | 91.46% |
| 50% | 50.00% | 91.73% |

The model retained good accuracy at all three filter-pruning levels.

The implementation used filter masks and did not physically rebuild the CNN with fewer channels. Therefore, actual hardware acceleration from channel removal was not established.

## 4.4 Pruning Graphs

The following plots were generated:

- `magnitude_pruning_accuracy.png`
- `structured_pruning_accuracy.png`

The magnitude pruning graph compares global and layer-wise pruning accuracy against sparsity.

The structured pruning graph presents test accuracy against filter sparsity.

# 5. Task 3 – Latency and Storage Comparison

The original dense baseline and the 95% globally pruned model were compared.

## 5.1 Storage Comparison

| Model representation | Storage |
|---|---:|
| Baseline dense | 3.8763 MB |
| 95% pruned dense | 3.8763 MB |
| 95% pruned sparse representation | 1.2339 MB |

The sparse representation reduced storage by 68.17%.

Unstructured pruning did not reduce the size of the dense checkpoint because the zero-valued weights were still stored.

## 5.2 Inference Latency

| Model | CPU latency |
|---|---:|
| Baseline dense | 1.8849 ms/image |
| 95% pruned dense | 2.4992 ms/image |

The pruned dense model was slower in this benchmark.

This demonstrates that unstructured pruning does not automatically improve inference speed. Standard dense convolution and linear operations continue processing dense tensors, including zero-valued weights.

The sparse representation was evaluated for storage, not inference latency.

# 6. Task 4 – Knowledge Distillation

Knowledge distillation was implemented using a large teacher CNN and a smaller student CNN.

The teacher model was the previously trained oversized CNN.

The student model contained 89,546 trainable parameters.

## 6.1 Distillation Configuration

| Parameter | Value |
|---|---:|
| Temperature | 4 |
| Alpha | 0.5 |
| Epochs | 5 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |

The combined loss function was:

L = alpha × CE + (1 - alpha) × T² × KL

The student was trained in two configurations:

1. Normal student training using classification labels.
2. Knowledge distillation using classification labels and teacher predictions.

## 6.2 Model Comparison

| Metric | Teacher | Normal Student | Distilled Student |
|---|---:|---:|---:|
| Parameters | 1,015,114 | 89,546 | 89,546 |
| Test accuracy | 90.60% | 88.59% | 88.10% |
| Model size | 3.8763 MB | 0.3455 MB | 0.3456 MB |
| CPU latency | 1.6641 ms | 0.7076 ms | 0.7055 ms |

The student model reduced the parameter count by approximately 91.18% compared with the teacher.

The normal student achieved slightly higher accuracy than the distilled student in the five-epoch experiment.

Knowledge distillation did not improve student accuracy in this particular experiment. However, both student models achieved substantially lower model size and inference latency than the teacher.

# 7. Task 5 – Efficiency Analysis

The experiments were evaluated using four efficiency pillars.

## 7.1 Data Efficiency

All experiments used Fashion-MNIST with the same preprocessing procedure and dataset distribution.

## 7.2 Model Efficiency

Global pruning maintained good accuracy at high sparsity levels.

Layer-wise pruning experienced accuracy degradation at high sparsity.

The student model achieved a 91.18% reduction in parameter count compared with the teacher.

## 7.3 Compute Efficiency

The student models achieved lower CPU inference latency than the teacher.

Unstructured pruning did not improve dense inference latency because the dense operations continued processing zero-valued weights.

## 7.4 Energy Efficiency

Direct energy consumption measurements were not performed because a dedicated power-monitoring instrument was unavailable.

However, the smaller student architecture and lower inference time suggest potential energy efficiency improvements.

These are indirect indicators, not measured energy savings.

# 8. Overall Conclusion

The laboratory demonstrated several approaches to neural network compression and their associated trade-offs.

Global magnitude pruning maintained good classification accuracy even at 95% sparsity. Layer-wise pruning was more sensitive to high pruning ratios.

Structured filter pruning retained good accuracy at 50% filter sparsity, although the implementation did not physically reduce the network's channel dimensions.

Knowledge distillation produced a compact student model containing approximately 91.18% fewer parameters than the teacher. The normal and distilled student models both achieved lower inference latency than the original teacher.

However, the normal student achieved slightly higher test accuracy than the distilled student after five epochs.

Overall, the experiments demonstrate that model compression should be evaluated using multiple factors, including accuracy, parameter count, storage requirements and inference latency.

A smaller model does not necessarily mean faster execution, and a high sparsity ratio does not automatically translate into hardware acceleration.