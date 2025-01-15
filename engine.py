import numpy as np


class Value:
    def __init__(
        self,
        data,
        label="",
        _children=(),
        _operation="",
    ):
        self.data = data
        self.label = label
        self.grad = 0

        self._children = set(_children)
        self._operation = _operation
        self._backward = lambda: None

    # ========== back-probgation ==========
    def backward(self):
        topo_nodes = []
        visited = set()

        def build_topo(node):
            if not node in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo_nodes.append(node)

        build_topo(self)
        for node in topo_nodes:
            node.grad = 0  # reset gradient

        self.grad = 1
        for node in reversed(topo_nodes):
            node._backward()

    # ========== activation functions ==========
    def tanh(self):
        x = self.data
        data = np.tanh(x)
        out = Value(data, _children=(self,), _operation="tanh")

        def _backward():
            self.grad += (1 - data**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        data = self.data if self.data > 0 else 0
        out = Value(data, _children=(self,), _operation="relu")

        def _backward():
            self.grad += out.grad * (1 if self.data > 0 else 0)

        out._backward = _backward
        return out

    # ========== math operations ==========
    def exp(self):
        data = np.exp(self.data)
        out = Value(data, _children=(self,), _operation="exp")

        def _backward():
            self.grad += out.grad * data

        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        data = self.data + other.data
        out = Value(data, _children=(self, other), _operation="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return self + -other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        data = self.data * other.data
        out = Value(data, _children=(self, other), _operation="*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "other must be an int or float"

        data = self.data**other
        out = Value(data, _children=(self,), _operation=f"{self.data}^{other}")

        def _backward():
            self.grad += out.grad * other * (self.data ** (other - 1))

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other**-1)

    # ========== internal representation ==========
    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"
