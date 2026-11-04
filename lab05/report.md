# Machine Learning -- Lab 5 Report

## Frameworks and Automatic Differentiation

**Student:** Jishnu Krishna Menon\
**Course:** Bachelor of Engineering in Artificial Intelligence\
**Subject:** Machine Learning\
**Lab:** 05 -- Frameworks and Automatic Differentiation

------------------------------------------------------------------------

## 1. Introduction

The aim of this laboratory was to explore automatic differentiation and
the role of machine-learning frameworks. The work included implementing
a scalar automatic-differentiation engine, validating its gradients
against PyTorch and numerical estimates, training a small neural network
on the `make_moons` dataset, and examining PyTorch execution modes and
tensor memory layout.

The experiments used PyTorch 2.3.1+cpu, NumPy 1.26.4 and scikit-learn
1.5.0. The environment used CPU execution; CUDA was unavailable.

## 2. Task 1 -- Design of the Value class

A `Value` class was implemented to represent a scalar node in a
computational graph. Each object stores: - `data`: the scalar numerical
value; - `grad`: the gradient associated with the node; - `_prev`:
references to parent nodes; - `_op`: the operation that created the
node; - `_backward`: a function defining how gradients are propagated to
parent nodes.

The initial tests successfully created values of 3 and 4. Their
gradients were initialized to 0.0.

## 3. Task 2 -- Mathematical operations

The `Value` class was extended with addition, multiplication, power and
ReLU. Each operation creates a new node, stores its parent nodes in the
computational graph, and defines a local backward rule using the chain
rule.

  Operation              Test   Result
  ---------------- ---------- --------
  Addition              3 + 4        7
  Multiplication        3 × 4       12
  Power                    3²        9
  ReLU                ReLU(5)        5
  ReLU               ReLU(-5)        0

## 4. Task 3 -- Topological backpropagation and gradient accumulation

A `backward()` method was implemented. It builds a topological ordering
of the graph using a depth-first traversal and a visited set. It then
initializes the output gradient to 1.0 and visits nodes in reverse
topological order, calling each node's backward function.

Gradient contributions are accumulated using `+=` rather than
overwritten. This is necessary when a variable contributes to more than
one path in the computational graph.

For the expression:

\[ f=(a `\times `{=tex}b+c)\^2 \]

with a=2, b=3, and c=1, the forward output was 49.

  Variable     Computed gradient
  ---------- -------------------
  a                           42
  b                           28
  c                           14

A separate accumulation test used f=a×a, with a=3. The output was 9 and
the gradient was 6, confirming that both contributions to the same
variable were accumulated.

## 5. Task 4 -- Gradient validation

The custom engine was validated using:

\[ f=((a `\times `{=tex}b+c)\^2).relu().sum() \]

with a=2, b=3, and c=1. Gradients were compared with PyTorch autograd
and central finite-difference numerical estimates. The numerical
estimates used a step size of ε=10⁻⁶.

  Variable     Custom engine   PyTorch autograd   Numerical estimate
  ---------- --------------- ------------------ --------------------
  a                     42.0               42.0    41.99999999698889
  b                     28.0               28.0   28.000000003913783
  c                     14.0               14.0   14.000000003733248

The maximum absolute difference across the comparisons was:

\[ 3.9137830754043534 `\times 10`{=tex}\^{-9} \]

The required tolerance was less than 10⁻⁶. The validation passed.

## 6. Task 5 -- Neural network training on make_moons

A small neural network was trained on a dataset generated using
scikit-learn's `make_moons`.

### Experiment configuration

  Setting                                Value
  ------------------- ------------------------
  Dataset                           make_moons
  Number of samples                        200
  Noise                                    0.2
  Random seed                               42
  Architecture                       2 → 8 → 1
  Hidden activation                       ReLU
  Output activation                    Sigmoid
  Loss                    Binary cross-entropy
  Optimizer             Gradient descent / SGD
  Learning rate                           0.05
  Epochs                                    20
  Timing runs                                5

The custom implementation used the scalar `Value` engine. A
corresponding PyTorch model used linear layers, ReLU, sigmoid, binary
cross-entropy loss and SGD.

### Timing results

           Run   Custom engine (seconds)   PyTorch (seconds)
  ------------ ------------------------- -------------------
             1                    3.7909              0.0530
             2                    4.2481              0.0128
             3                    4.3277              0.0131
             4                    4.3654              0.0139
             5                    4.3188              0.0132
    **Median**                **4.3188**          **0.0132**

Based on the measured medians, PyTorch was approximately 326.28 times
faster in this experiment.

The final loss values printed by the last run were 0.700392 for the
custom engine and 0.624600 for PyTorch. These are last-run loss values,
not median losses. The implementations use different parameter
initialization and execution details, so this timing experiment should
not be interpreted as a controlled comparison of model quality.

The large timing difference is consistent with the implementation
approaches: the custom engine performs scalar operations through Python
objects, while PyTorch uses optimized tensor operations and numerical
libraries.

## 7. Task 6 -- Framework execution and tensor properties

### 7.1 Computational graph inspection

A small PyTorch model with architecture 2 → 8 → 1 was traced using
`torch.jit.trace`. The printed graph showed the first linear layer, ReLU
activation, second linear layer and sigmoid activation.

### 7.2 Eager and traced execution

The environment used PyTorch 2.3.1+cpu. The experiment attempted
`torch.compile()`, but compilation failed with a Windows temporary-path
`unicodeescape` syntax error while importing a generated TorchInductor
module. As allowed by the lab instructions, `torch.jit.trace()` was used
as a fallback.

  Execution mode         Median time (seconds)
  -------------------- -----------------------
  Eager                             0.00004030
  TorchScript traced                0.00002570

For this small model and this particular measurement, traced execution
was approximately 1.57 times faster than eager execution. This is an
observed result for this setup; it does not establish that tracing will
always be faster. The model is small and the absolute timings are very
short.

### 7.3 Tensor shape, dtype, contiguity and stride

A 512 × 512 tensor was inspected, transposed, and then converted to a
contiguous tensor.

  Tensor            Shape       Data type   Contiguous   Stride
  ----------------- ----------- ----------- ------------ ----------
  Original          512 × 512   float32     Yes          (512, 1)
  Transposed view   512 × 512   float32     No           (1, 512)
  Contiguous copy   512 × 512   float32     Yes          (512, 1)

Transposing the tensor produced a non-contiguous view with a different
stride. Calling `.contiguous()` created a tensor with the standard
contiguous memory layout.

### 7.4 Contiguous and non-contiguous operation timing

The same element-wise operation, `x * 1.5 + 2.0`, was benchmarked on
both tensor layouts.

  Tensor layout      Median time (seconds)
  ---------------- -----------------------
  Non-contiguous                0.00003915
  Contiguous                    0.00003940

The measured difference was very small. For this operation, tensor size
and CPU environment, the experiment did not show a meaningful
performance penalty for the non-contiguous view. Different operations,
tensor sizes and hardware may produce different results.

## 8. Overall observations

1.  A computational graph can store the relationships between scalar
    operations.
2.  The chain rule can be implemented by defining local backward
    functions for each operation.
3.  Topological ordering allows gradients to be propagated in the
    correct order.
4.  Gradient accumulation is essential when a variable is used along
    multiple paths.
5.  The custom gradients agreed with PyTorch autograd and numerical
    estimates within the required tolerance.
6.  The custom scalar engine was much slower than PyTorch for the tested
    neural-network workload.
7.  Eager and traced execution can have different performance, but the
    outcome depends on the model and environment.
8.  Tensor strides and contiguity describe memory layout and can affect
    performance depending on the operation.

## 9. Conclusion

All six laboratory tasks were implemented and executed. The custom
scalar automatic-differentiation engine supported the required
operations and topological backpropagation. Its gradients passed
validation against PyTorch and numerical estimates, with a maximum
absolute difference of approximately 3.91 × 10⁻⁹, below the required
10⁻⁶ tolerance.

The `make_moons` experiment showed a substantial timing advantage for
PyTorch over the scalar Python implementation. The final task
demonstrated how to inspect a traced computational graph, compare eager
and traced execution, and examine tensor contiguity and stride.

The experiments were performed on a CPU-only PyTorch installation.
`torch.compile()` could not be used successfully in this environment, so
TorchScript tracing was used as the fallback. These environment and
measurement limitations should be considered when interpreting the
benchmark results.
