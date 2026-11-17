# Lab 6: Training Pipelines and Project Proposal

**Student Name:** Jishnu Krishna Menon  
**Student ID:** 72987  
**University:** Vistula University  
**Programme:** Bachelor of Engineering in Artificial Intelligence  
**Subject:** Machine Learning

---

## 1. Introduction

The objective of this laboratory was to develop a reproducible machine learning training pipeline using PyTorch and YAML configuration. The experiments focused on model training, experiment tracking, performance profiling, data loading optimization, batch-size comparison, optimizer and scheduler evaluation, gradient accumulation, and checkpoint recovery.

The Fashion-MNIST dataset and a Convolutional Neural Network (CNN) from Lab 3 were used as the baseline.

## 2. Experimental Environment

| Component | Configuration |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch 2.3.1 |
| Dataset | Fashion-MNIST |
| Model | CNN |
| Experiment tracking | MLflow 3.18.0 |
| Configuration | YAML |
| Hardware | CPU |
| CUDA | Not available |
| Random seed | 42 |

The dataset was divided into 55,000 training images, 5,000 validation images, and 10,000 test images.

## 3. Task 1 – Reproducible Training Pipeline

The original CNN from Lab 3 was integrated into a YAML-driven training pipeline.

The configuration file contains the model settings, dataset parameters, batch size, learning rate, optimizer, scheduler, random seed, and checkpoint settings.

MLflow was integrated to record experiment parameters, training metrics, validation metrics, and model artifacts.

The baseline training configuration was:

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Epochs: 5
- Random seed: 42

### Baseline Results

| Epoch | Training Loss | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|
| 1 | 0.5347 | 0.3904 | 86.38% |
| 2 | 0.3503 | 0.3360 | 87.90% |
| 3 | 0.3038 | 0.3119 | 89.18% |
| 4 | 0.2802 | 0.3143 | 88.70% |
| 5 | 0.2633 | 0.2793 | 89.78% |

**Best validation accuracy:** 89.78%  
**Test accuracy:** 89.53%

The training pipeline successfully saved model checkpoints and logged the experiment using MLflow.

## 4. Task 2 – Training Pipeline Profiling

PyTorch Profiler was used to examine the execution time of one training epoch.

The measured execution-time breakdown was:

| Operation | Time (seconds) | Percentage |
|---|---:|---:|
| Data loading | 30.3499 | 60.68% |
| Forward pass | 8.2078 | 16.41% |
| Backward pass | 10.5481 | 21.09% |
| Optimizer step | 0.9086 | 1.82% |

### Analysis

Data loading consumed the largest share of the measured training time, accounting for approximately 60.68%.

The backward pass required more time than the forward pass because it calculates gradients for updating the model parameters.

The optimizer step consumed the smallest share of the measured time.

This indicates that data loading was the main bottleneck in the measured training pipeline.

The timing breakdown is based on manually measured stage timings. The overall profiler duration was higher because profiling introduces additional overhead.

**Figure:** `plots/profiler_breakdown.png`

## 5. Task 3 – DataLoader Worker Comparison

The number of DataLoader workers was varied using the values 0, 1, 2, 4, and 8.

The experiment used a fixed subset of 10,000 training images and a batch size of 128.

### Results

| Workers | Time (seconds) | Throughput (images/second) |
|---:|---:|---:|
| 0 | 2.6812 | 3729.72 |
| 1 | 11.4756 | 871.42 |
| 2 | 17.3603 | 576.03 |
| 4 | 32.1206 | 311.33 |
| 8 | 62.5706 | 159.82 |

### Analysis

The experiment showed that using zero workers provided the highest throughput.

Increasing the number of workers reduced performance in this CPU-based experiment. This may be due to process creation, inter-process communication, and scheduling overhead.

For this lightweight Fashion-MNIST dataset and the current environment, using `num_workers=0` was the most efficient configuration.

**Figure:** `plots/worker_throughput.png`

## 6. Task 4 – Batch Size Comparison

Five different batch sizes were tested: 32, 64, 128, 256, and 512.

Each configuration was trained for five epochs.

### Results

| Batch Size | Training Time (seconds) | Throughput (images/second) | Peak Memory (MB) | Validation Accuracy |
|---:|---:|---:|---:|---:|
| 32 | 201.35 | 1365.79 | 324.41 | 90.30% |
| 64 | 182.57 | 1506.30 | 330.02 | 90.04% |
| 128 | 173.49 | 1585.11 | 348.89 | 89.30% |
| 256 | 174.22 | 1578.48 | 378.67 | 89.10% |
| 512 | 166.86 | 1648.10 | 420.07 | 87.66% |

### Analysis

Batch size 512 achieved the highest throughput and shortest training time.

However, it also required the highest memory usage and produced the lowest validation accuracy among the tested configurations.

Batch size 32 achieved the highest validation accuracy and lowest memory usage.

The experiment demonstrates a trade-off between throughput, memory consumption, and model performance.

The memory measurements represent total process peak resident memory (RSS), not GPU memory.

**Figures:**
- `plots/batch_throughput.png`
- `plots/batch_memory.png`
- `plots/batch_accuracy.png`

## 7. Task 5 – Optimizer and Learning Rate Scheduler Comparison

Two optimizers, Adam and SGD with momentum, were evaluated using three learning-rate schedulers:

- StepLR
- CosineAnnealingLR
- ReduceLROnPlateau

Each combination was trained for ten epochs using batch size 128.

### Results

| Optimizer | Scheduler | Best Validation Accuracy | Best Epoch |
|---|---|---:|---:|
| Adam | StepLR | 89.80% | 9 |
| Adam | CosineAnnealingLR | 90.68% | 10 |
| Adam | ReduceLROnPlateau | 91.00% | 10 |
| SGD | StepLR | 85.00% | 7 |
| SGD | CosineAnnealingLR | 86.24% | 10 |
| SGD | ReduceLROnPlateau | 86.64% | 10 |

### Analysis

Adam performed better than SGD with momentum across all three scheduler configurations.

Adam combined with ReduceLROnPlateau achieved the highest validation accuracy of 91.00%.

This experiment indicates that optimizer and scheduler selection can significantly influence model performance.

**Figure:** `plots/optimizer_scheduler_comparison.png`

## 8. Task 6 – Gradient Accumulation

Gradient accumulation was implemented to simulate an effective batch size of 256 using four micro-batches of 64.

The two approaches were compared over five epochs.

### Results

| Epoch | Batch 256 Accuracy | Batch 64 + Accumulation 4 Accuracy |
|---:|---:|---:|
| 1 | 85.04% | 84.92% |
| 2 | 86.66% | 86.74% |
| 3 | 87.26% | 87.32% |
| 4 | 87.94% | 87.96% |
| 5 | 88.90% | 88.88% |

Training time:

- Batch 256: 180.12 seconds
- Batch 64 with accumulation 4: 186.93 seconds

### Analysis

Both approaches produced very similar training behaviour.

The final validation accuracy differed by only 0.02 percentage points.

Gradient accumulation allows a larger effective batch size to be simulated using smaller micro-batches, with an optimizer update performed after accumulating gradients.

**Figure:** `plots/gradient_accumulation_comparison.png`

## 9. Task 7 – Checkpoint Saving and Training Recovery

Training was intentionally interrupted after epoch 3.

The latest checkpoint stored the model state, optimizer state, scheduler state, training history, and random number generator states.

Training was then resumed using the saved checkpoint.

### Results

The resumed training continued from epoch 4.

| Epoch | Validation Accuracy |
|---:|---:|
| 1 | 86.38% |
| 2 | 87.90% |
| 3 | 89.18% |
| 4 | 88.70% |
| 5 | 89.78% |

The resumed epoch 4 and epoch 5 results matched the uninterrupted baseline experiment.

### Analysis

The experiment confirmed that the training pipeline could recover from an interruption using saved checkpoints.

This demonstrates the importance of checkpointing for reproducibility and reliable machine learning workflows.

## 10. Conclusion

This laboratory demonstrated the development of a reproducible machine learning training pipeline using PyTorch, YAML, and MLflow.

The experiments identified data loading as the main measured bottleneck and showed that increasing the number of DataLoader workers did not improve performance in the tested CPU environment.

The batch-size comparison demonstrated a trade-off between training throughput, memory usage, and validation accuracy.

Adam with ReduceLROnPlateau achieved the best validation accuracy among the tested optimizer and scheduler combinations.

Gradient accumulation successfully simulated a larger effective batch size while producing similar validation results.

Finally, checkpoint recovery demonstrated that interrupted training could be resumed successfully.

Overall, the laboratory provided practical experience in training pipeline design, profiling, optimization, experiment tracking, and reproducibility.

## 11. Individual Project Proposal

The individual project proposal is provided separately in `proposal.md`.

**Project:** Lightweight Fashion Image Classification for Edge Devices.

The project aims to investigate pruning, quantization, and knowledge distillation to reduce model size and inference latency while maintaining classification performance.

The deployment targets are a model size of no more than 5 MB and inference latency of no more than 50 milliseconds.

The proposal is subject to instructor approval.