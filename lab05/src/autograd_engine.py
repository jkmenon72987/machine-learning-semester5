import math


class Value:

    def __init__(self, data, _children=(), _op=''):

        self.data = float(data)
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op

        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # Addition
    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # Negation
    def __neg__(self):
        return self * -1

    # Subtraction
    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -Value(other))

    def __rsub__(self, other):
        return Value(other) - self

    # Multiplication
    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # Division
    def __truediv__(self, other):

        other = other if isinstance(other, Value) else Value(other)

        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Value(other) / self

    # Power
    def __pow__(self, power):

        assert isinstance(power, (int, float))

        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += power * self.data ** (power - 1) * out.grad

        out._backward = _backward
        return out

    # Exponential
    def exp(self):

        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    # Natural logarithm
    def log(self):

        if self.data <= 0:
            raise ValueError("Logarithm input must be positive")

        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # Sigmoid
    def sigmoid(self):

        value = 1.0 / (1.0 + math.exp(-self.data))

        out = Value(value, (self,), 'sigmoid')

        def _backward():
            self.grad += (
                out.data * (1.0 - out.data) * out.grad
            )

        out._backward = _backward
        return out

    # ReLU
    def relu(self):

        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (
                (1.0 if self.data > 0 else 0.0) * out.grad
            )

        out._backward = _backward
        return out

    # Sum
    def sum(self):

        out = Value(self.data, (self,), 'sum')

        def _backward():
            self.grad += out.grad

        out._backward = _backward
        return out

    # Backward pass
    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):

            if v not in visited:

                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()