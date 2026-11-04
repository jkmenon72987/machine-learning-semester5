import torch

from lab05.src.autograd_engine import Value


# =====================================
# TASK 4: GRADIENT VALIDATION
# =====================================

a_value = 2.0
b_value = 3.0
c_value = 1.0


# -------------------------------------
# 1. Custom automatic differentiation
# -------------------------------------

a = Value(a_value)
b = Value(b_value)
c = Value(c_value)

custom_output = ((a * b + c) ** 2).relu().sum()

custom_output.backward()

custom_gradients = [
    a.grad,
    b.grad,
    c.grad
]

print("\nCUSTOM AUTOGRAD RESULTS")
print("-----------------------")

print("Output:", custom_output.data)
print("Gradient a:", a.grad)
print("Gradient b:", b.grad)
print("Gradient c:", c.grad)


# -------------------------------------
# 2. PyTorch automatic differentiation
# -------------------------------------

ta = torch.tensor(a_value, requires_grad=True)
tb = torch.tensor(b_value, requires_grad=True)
tc = torch.tensor(c_value, requires_grad=True)

torch_output = torch.relu((ta * tb + tc) ** 2).sum()

torch_output.backward()

torch_gradients = [
    ta.grad.item(),
    tb.grad.item(),
    tc.grad.item()
]

print("\nPYTORCH AUTOGRAD RESULTS")
print("------------------------")

print("Output:", torch_output.item())
print("Gradient a:", ta.grad.item())
print("Gradient b:", tb.grad.item())
print("Gradient c:", tc.grad.item())


# -------------------------------------
# 3. Numerical gradient calculation
# -------------------------------------

def calculate_function(a, b, c):

    value = (a * b + c) ** 2

    return max(0.0, value)


def numerical_gradient(variable, a, b, c, epsilon=1e-6):

    if variable == "a":

        plus = calculate_function(a + epsilon, b, c)
        minus = calculate_function(a - epsilon, b, c)

    elif variable == "b":

        plus = calculate_function(a, b + epsilon, c)
        minus = calculate_function(a, b - epsilon, c)

    elif variable == "c":

        plus = calculate_function(a, b, c + epsilon)
        minus = calculate_function(a, b, c - epsilon)

    return (plus - minus) / (2 * epsilon)


numerical_gradients = [
    numerical_gradient("a", a_value, b_value, c_value),
    numerical_gradient("b", a_value, b_value, c_value),
    numerical_gradient("c", a_value, b_value, c_value)
]

print("\nNUMERICAL GRADIENT RESULTS")
print("-------------------------")

print("Gradient a:", numerical_gradients[0])
print("Gradient b:", numerical_gradients[1])
print("Gradient c:", numerical_gradients[2])


# -------------------------------------
# 4. Compare all gradients
# -------------------------------------

names = ["a", "b", "c"]

print("\nGRADIENT COMPARISON")
print("-------------------")

for i, name in enumerate(names):

    print(f"\nVariable: {name}")

    print("Custom:", custom_gradients[i])
    print("PyTorch:", torch_gradients[i])
    print("Numerical:", numerical_gradients[i])


# -------------------------------------
# 5. Final validation
# -------------------------------------

differences = []

for i in range(3):

    differences.append(
        abs(custom_gradients[i] - torch_gradients[i])
    )

    differences.append(
        abs(custom_gradients[i] - numerical_gradients[i])
    )

max_difference = max(differences)

print("\nFINAL VALIDATION")
print("----------------")

print("Maximum absolute difference:", max_difference)

if max_difference < 1e-6:

    print("PASS: All gradients match within tolerance.")

else:

    print("FAIL: Gradient difference exceeds tolerance.")