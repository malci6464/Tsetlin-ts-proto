"""Minimal drop-in replacement for ``pyTsetlinMachine.tm`` used in examples.

This fallback implementation does *not* implement a real Tsetlin Machine. It
approximates the intended interface so that the example script can be executed
in environments where the official ``pyTsetlinMachine`` package is not
available (for instance, offline evaluation environments).

The classifier provided here is a simple multi-class perceptron operating on
binary feature vectors. Although behaviourally different from a true Tsetlin
Machine, it exposes the subset of the public API that the example exercises:

* ``MultiClassTsetlinMachine`` with the signature ``(n_clauses, T, s)``
* ``fit`` supporting ``epochs`` and ``incremental`` keyword arguments
* ``predict`` returning the most likely class indices for the inputs

The goal is simply to provide a deterministic, lightweight alternative so that
unit tests and documentation examples continue to run without network access.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class _PerceptronState:
    """Internal weight matrix for the surrogate classifier."""

    weights: List[List[float]]
    bias: List[float]


class MultiClassTsetlinMachine:
    """Small perceptron-based surrogate for the true Tsetlin Machine.

    Parameters
    ----------
    n_clauses:
        Unused placeholder to match the real constructor. Kept for API
        compatibility.
    T:
        Voting threshold placeholder retained for signature compatibility.
    s:
        Specificity parameter placeholder retained for signature compatibility.
    learning_rate:
        Optional perceptron learning rate. Defaults to ``0.1`` which performs
        well for the synthetic datasets shipped in the repository.
    random_state:
        Optional seed for the RNG controlling weight initialisation and sample
        shuffling.
    """

    def __init__(
        self,
        n_clauses: int,
        T: int,
        s: float,
        *,
        learning_rate: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_clauses = n_clauses
        self.T = T
        self.s = s
        self.learning_rate = learning_rate
        self.random_state = random_state

        self._state: Optional[_PerceptronState] = None
        self._rng = random.Random(random_state)

    def _initialise_state(self, n_features: int, n_classes: int) -> None:
        scale = 1.0 / max(1, n_features)
        weights = [
            [self._rng.uniform(-scale, scale) for _ in range(n_features)]
            for _ in range(n_classes)
        ]
        bias = [0.0 for _ in range(n_classes)]
        self._state = _PerceptronState(weights=weights, bias=bias)

    def fit(
        self,
        X: Sequence[Sequence[int]],
        y: Sequence[int],
        *,
        epochs: int = 1,
        incremental: bool = False,
    ) -> "MultiClassTsetlinMachine":
        if not X:
            raise ValueError("Expected non-empty training data")
        if len(X) != len(y):
            raise ValueError("Mismatched number of samples between X and y")

        n_samples = len(X)
        n_features = len(X[0])
        classes = sorted({int(label) for label in y})
        n_classes = len(classes)

        if self._state is None or not incremental:
            self._initialise_state(n_features, n_classes)

        assert self._state is not None

        for _ in range(max(1, epochs)):
            indices = list(range(n_samples))
            self._rng.shuffle(indices)
            for idx in indices:
                xi = X[idx]
                target = int(y[idx])
                activations = self._activations(xi)
                predicted = activations.index(max(activations))
                if predicted != target:
                    self._apply_update(xi, target, predicted)

        return self

    def predict(self, X: Sequence[Sequence[int]]) -> List[int]:
        if self._state is None:
            raise RuntimeError("The model has not been fitted yet.")

        predictions: List[int] = []
        for xi in X:
            activations = self._activations(xi)
            predictions.append(activations.index(max(activations)))
        return predictions

    def _activations(self, features: Sequence[int]) -> List[float]:
        assert self._state is not None
        scores: List[float] = []
        for weight_vector, bias in zip(self._state.weights, self._state.bias):
            activation = bias
            for w, x in zip(weight_vector, features):
                activation += w * float(x)
            scores.append(activation)
        return scores

    def _apply_update(self, features: Sequence[int], target: int, predicted: int) -> None:
        assert self._state is not None
        for index, value in enumerate(features):
            update = self.learning_rate * float(value)
            self._state.weights[target][index] += update
            self._state.weights[predicted][index] -= update
        self._state.bias[target] += self.learning_rate
        self._state.bias[predicted] -= self.learning_rate
