"""Small NumPy reverse-mode differentiation and Adam for the CPU learner.

Only operations needed by generals_policy are implemented. No dependency is
downloaded. Parameters and intermediates use float64 for gradient checking.
"""
from __future__ import annotations

import numpy as np
from contextlib import contextmanager
from contextvars import ContextVar

_grad_enabled = ContextVar('generals_grad_enabled', default=True)


@contextmanager
def no_grad():
    """Avoid constructing backward graphs during collection and evaluation."""
    token = _grad_enabled.set(False)
    try:
        yield
    finally:
        _grad_enabled.reset(token)


def unbroadcast(gradient, shape):
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    def __init__(self, data, parents=(), backward=None, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.parents = tuple(p for p in parents if p.requires_grad) if _grad_enabled.get() else ()
        self.requires_grad = _grad_enabled.get() and (requires_grad or bool(self.parents))
        self.grad = None
        self._backward = backward if self.requires_grad else None

    def add_grad(self, value):
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.asarray(value, dtype=np.float64).copy()
            else:
                self.grad += value

    def __add__(self, other):
        other = tensor(other)
        def backward(g):
            self.add_grad(unbroadcast(g, self.data.shape))
            other.add_grad(unbroadcast(g, other.data.shape))
        return Tensor(self.data + other.data, (self, other), backward)

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -tensor(other)

    def __rsub__(self, other):
        return tensor(other) + -self

    def __mul__(self, other):
        other = tensor(other)
        def backward(g):
            self.add_grad(unbroadcast(g * other.data, self.data.shape))
            other.add_grad(unbroadcast(g * self.data, other.data.shape))
        return Tensor(self.data * other.data, (self, other), backward)

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (1.0 / scalar)

    def __matmul__(self, other):
        other = tensor(other)
        a, b = self.data, other.data
        if a.ndim not in (1, 2) or b.ndim not in (1, 2):
            raise ValueError('Only vector/matrix products are supported')
        def backward(g):
            if a.ndim == b.ndim == 1:
                da, db = g * b, g * a
            elif a.ndim == 1:
                da, db = g @ b.T, np.outer(a, g)
            elif b.ndim == 1:
                da, db = np.outer(g, b), a.T @ g
            else:
                da, db = g @ b.T, a.T @ g
            self.add_grad(da)
            other.add_grad(db)
        return Tensor(a @ b, (self, other), backward)

    def __getitem__(self, index):
        # Pointer selections grow while constructing an action. Backpropagation
        # must use the indices from this forward operation, not their later value.
        def snapshot(value):
            if isinstance(value, (list, np.ndarray)):
                return np.asarray(value).copy()
            if isinstance(value, tuple):
                return tuple(snapshot(item) for item in value)
            return value
        index = snapshot(index)
        def backward(g):
            if self.requires_grad:
                gradient = np.zeros_like(self.data)
                np.add.at(gradient, index, g)
                self.add_grad(gradient)
        return Tensor(self.data[index], (self,), backward)

    def tanh(self):
        result = np.tanh(self.data)
        return Tensor(result, (self,), lambda g: self.add_grad(g * (1 - result * result)))

    def sigmoid(self):
        # tanh form avoids overflow for large positive/negative inputs.
        result = (1 + np.tanh(self.data / 2)) / 2
        return Tensor(result, (self,), lambda g: self.add_grad(g * result * (1 - result)))

    def exp(self):
        result = np.exp(self.data)
        return Tensor(result, (self,), lambda g: self.add_grad(g * result))

    def clip(self, low, high):
        result = np.clip(self.data, low, high)
        return Tensor(result, (self,), lambda g: self.add_grad(g * ((self.data > low) & (self.data < high))))

    def transpose(self):
        return Tensor(self.data.T, (self,), lambda g: self.add_grad(g.T))

    def reshape(self, shape):
        return Tensor(self.data.reshape(shape), (self,), lambda g: self.add_grad(g.reshape(self.data.shape)))

    def softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        result = np.exp(shifted)
        result /= result.sum(axis=axis, keepdims=True)
        return Tensor(result, (self,), lambda g: self.add_grad(
            result * (g - (g * result).sum(axis=axis, keepdims=True))))

    def layer_norm(self, epsilon=1e-5):
        centered = self.data - self.data.mean(axis=-1, keepdims=True)
        inv = 1 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + epsilon)
        result = centered * inv
        def backward(g):
            self.add_grad(inv * (g - g.mean(axis=-1, keepdims=True)
                                - result * (g * result).mean(axis=-1, keepdims=True)))
        return Tensor(result, (self,), backward)

    def sum(self, axis=None):
        def backward(g):
            if axis is not None:
                g = np.expand_dims(g, axis)
            self.add_grad(np.broadcast_to(g, self.data.shape))
        return Tensor(self.data.sum(axis=axis), (self,), backward)

    def mean(self, axis=None):
        count = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis) / count

    def detach(self):
        return Tensor(self.data.copy())

    def backward(self):
        if self.data.size != 1:
            raise ValueError('backward requires a scalar loss')
        ordered, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                ordered.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in reversed(node.parents))
        # Parameters can appear in multiple graphs; retain their accumulated
        # gradients, but clear intermediates when backward is called again.
        for node in ordered:
            if node.parents:
                node.grad = None
        self.grad = np.ones_like(self.data)
        for node in reversed(ordered):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)


def tensor(value):
    return value if isinstance(value, Tensor) else Tensor(value)


def concatenate(values):
    values = [tensor(v) for v in values]
    sizes = [v.data.shape[0] for v in values]
    def backward(g):
        offset = 0
        for value, size in zip(values, sizes):
            value.add_grad(g[offset:offset + size])
            offset += size
    return Tensor(np.concatenate([v.data for v in values]), values, backward)


def minimum(first, second):
    first, second = tensor(first), tensor(second)
    choose_first = first.data <= second.data
    def backward(g):
        first.add_grad(unbroadcast(g * choose_first, first.data.shape))
        second.add_grad(unbroadcast(g * ~choose_first, second.data.shape))
    return Tensor(np.minimum(first.data, second.data), (first, second), backward)


def scatter_mean(values, indices, size):
    """Entity embeddings -> map cells, without an expensive dense scatter matrix."""
    values = tensor(values)
    indices = np.asarray(indices, dtype=int).copy()
    counts = np.maximum(np.bincount(indices, minlength=size), 1)
    result = np.zeros((size, values.data.shape[1]))
    np.add.at(result, indices, values.data)
    result /= counts[:, None]
    return Tensor(result, (values,), lambda g: values.add_grad(g[indices] / counts[indices, None]))


def conv2d_same(values, kernel, side):
    """A shared 3x3 convolution on flattened HWC cells, with zero padding."""
    values, kernel = tensor(values), tensor(kernel)
    channels = values.data.shape[1]
    if values.data.shape[0] != side * side or kernel.data.shape[0] != 9 * channels:
        raise ValueError('Invalid convolution dimensions')
    padded = np.pad(values.data.reshape(side, side, channels), ((1, 1), (1, 1), (0, 0)))
    patches = np.concatenate([padded[y:y+side, x:x+side].reshape(side * side, channels)
                              for y in range(3) for x in range(3)], axis=1)
    def backward(g):
        kernel.add_grad(patches.T @ g)
        if values.requires_grad:
            dpatch = g @ kernel.data.T
            gradient = np.zeros_like(padded)
            for ordinal, (y, x) in enumerate((y, x) for y in range(3) for x in range(3)):
                gradient[y:y+side, x:x+side] += dpatch[:, ordinal*channels:(ordinal+1)*channels].reshape(side, side, channels)
            values.add_grad(gradient[1:-1, 1:-1].reshape(values.data.shape))
    return Tensor(patches @ kernel.data, (values, kernel), backward)


def categorical(logits, mask, *, choice=None, rng=None, deterministic=False):
    """A masked categorical draw, its differentiable log probability and entropy."""
    mask = np.asarray(mask, dtype=bool)
    if logits.data.ndim != 1 or mask.shape != logits.data.shape or not mask.any():
        raise ValueError('Invalid categorical mask')
    shifted = np.where(mask, logits.data, -np.inf)
    shifted -= np.max(shifted)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    log_probs = np.zeros_like(probabilities)
    log_probs[mask] = shifted[mask] - np.log(np.exp(shifted).sum())
    if choice is None:
        choice = int(np.argmax(probabilities) if deterministic else rng.choice(len(mask), p=probabilities))
    if not 0 <= choice < len(mask) or not mask[choice]:
        raise ValueError('Target outside categorical mask')
    gradient = -probabilities.copy()
    gradient[choice] += 1
    logp = Tensor(log_probs[choice], (logits,), lambda g: logits.add_grad(g * gradient))
    entropy_value = -np.sum(probabilities * log_probs)
    entropy_gradient = -probabilities * (log_probs + entropy_value)
    entropy = Tensor(entropy_value, (logits,), lambda g: logits.add_grad(g * entropy_gradient))
    return choice, logp, entropy


class Adam:
    def __init__(self, parameters, lr=3e-4):
        self.parameters = parameters
        self.lr, self.step_count = lr, 0
        self.m = {k: np.zeros_like(p.data) for k, p in parameters.items()}
        self.v = {k: np.zeros_like(p.data) for k, p in parameters.items()}

    def zero_grad(self):
        for parameter in self.parameters.values():
            parameter.grad = None

    def step(self, *, scale=1, max_norm=5):
        gradients = {k: np.zeros_like(p.data) if p.grad is None else p.grad * scale
                     for k, p in self.parameters.items()}
        norm = float(np.sqrt(sum(np.sum(g * g) for g in gradients.values())))
        if not np.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient; checkpoint unchanged')
        factor = min(1, max_norm / max(norm, 1e-12))
        self.step_count += 1
        for key, p in self.parameters.items():
            gradient = gradients[key] * factor
            self.m[key] = .9 * self.m[key] + .1 * gradient
            self.v[key] = .999 * self.v[key] + .001 * gradient * gradient
            m = self.m[key] / (1 - .9 ** self.step_count)
            v = self.v[key] / (1 - .999 ** self.step_count)
            p.data -= self.lr * m / (np.sqrt(v) + 1e-8)
        return norm
