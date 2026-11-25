# Lab 7 – Efficiency Comparison

## 1. Data Efficiency

All experiments used the Fashion-MNIST dataset.

| Dataset property | Value |
|---|---|
| Training samples | 55,000 |
| Validation samples | 5,000 |
| Testing samples | 10,000 |
| Image dimensions | 28 × 28 |
| Number of classes | 10 |
| Batch size | 128 |

The same dataset and preprocessing procedure were used across the baseline, pruning and knowledge distillation experiments.

## 2. Model Efficiency

The oversized baseline CNN contained 1,015,114 parameters and achieved 90.60% test accuracy.

### Magnitude pruning

Global pruning maintained 91.03% accuracy even at 95% weight sparsity.

Layer-wise pruning performed well up to 75% sparsity but experienced accuracy degradation at higher sparsity levels. At 95% sparsity, accuracy decreased to 82.88%.

### Structured pruning

Structured L2 filter pruning achieved:

| Filter sparsity | Test accuracy |
|---|---:|
| 10% | 91.94% |
| 25% | 91.46% |
| 50% | 91.73% |

### Knowledge distillation

| Model | Parameters | Test accuracy | Size |
|---|---:|---:|---:|
| Teacher CNN | 1,015,114 | 90.60% | 3.8763 MB |
| Normal Student | 89,546 | 88.59% | 0.3455 MB |
| Distilled Student | 89,546 | 88.10% | 0.3456 MB |

The student model reduced the parameter count by approximately 91.18%.

The normal student achieved slightly higher accuracy than the distilled student in this experiment.

## 3. Compute Efficiency

Inference latency was measured on the CPU.

### Dense and sparse comparison

| Model representation | Storage | Inference latency |
|---|---:|---:|
| Baseline dense | 3.8763 MB | 1.8849 ms |
| 95% pruned dense | 3.8763 MB | 2.4992 ms |
| 95% pruned sparse representation | 1.2339 MB | Not benchmarked |

The sparse representation reduced storage by 68.17%.

However, unstructured pruning did not improve inference latency because standard dense operations continued processing dense tensors containing zero-valued weights.

### Knowledge distillation comparison

| Model | Inference latency |
|---|---:|
| Teacher CNN | 1.6641 ms |
| Normal Student | 0.7076 ms |
| Distilled Student | 0.7055 ms |

The smaller student architecture achieved lower inference latency than the teacher.

## 4. Energy Efficiency

Direct energy consumption measurements were not performed because a dedicated power-monitoring instrument was not available.

However, the experiments provide some indirect indicators of potential energy efficiency:

- The student models contain substantially fewer parameters.
- Student inference required less measured CPU time per image.
- Sparse storage reduced the amount of model data stored on disk.
- The smaller student architecture offers a potential reduction in computational requirements.

These observations suggest potential energy savings, but actual energy consumption was not measured and therefore cannot be quantified from these experiments.

## 5. Overall Comparison

| Efficiency pillar | Main observation |
|---|---|
| Data | All experiments used the same Fashion-MNIST dataset and preprocessing. |
| Model | Global pruning retained accuracy at high sparsity; the student model reduced parameters by 91.18%. |
| Compute | The student model achieved lower CPU inference latency. Unstructured pruning did not improve dense inference speed. |
| Energy | Potential savings were identified through reduced model size and inference time, but direct energy measurements were unavailable. |

## 6. Conclusion

The experiments demonstrate that model compression involves trade-offs between model size, accuracy and computational efficiency.

Global magnitude pruning maintained good classification accuracy even at high sparsity levels, while layer-wise pruning showed greater sensitivity at 90% and 95% sparsity.

Structured filter pruning retained good accuracy at 50% filter sparsity. However, the implementation used filter masks rather than physically rebuilding the network with fewer channels, so actual hardware acceleration was not established.

Knowledge distillation produced a student model with approximately 91.18% fewer parameters than the teacher. The student models also achieved lower inference latency. However, the normal student achieved slightly higher accuracy than the distilled student after five epochs.

Overall, the smaller student architecture provided the clearest measured improvement in inference efficiency, while global pruning demonstrated that high weight sparsity can be achieved without necessarily sacrificing classification accuracy.

The experiments highlight the importance of evaluating model compression using accuracy, storage, inference latency and energy considerations rather than relying on parameter count or sparsity alone.