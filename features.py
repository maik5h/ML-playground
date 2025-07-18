import numpy as np
from typing import List


class Feature:
    """
    Base class for features used for type hints.
    """
    pass

class PolynomialFeature(Feature):
    """
    Polynomial feature with an integer power.
    """
    def __init__(self, power: int):
        self.power = power
        self.name = f'$phi(x) = x^{power}$'

    def __call__(self, input: np.array):
        return input ** self.power

class SineFeature(Feature):
    """
    Sine feature with frequency of a natural multiple of pi.
    """
    def __init__(self, n_pi: int):
        self.n = n_pi
        self.name = f'phi(x) = sin({n_pi:.2f}\pi x)'

    def __call__(self, input: np.array):
        return np.sin(self.n * np.pi * input)

class CosineFeature(Feature):
    """
    Cosine feature with frequency of a natural multiple of pi.
    """
    def __init__(self, n_pi: int):
        self.n = n_pi
        self.name = f'phi(x) = cos({n_pi:.2f}\pi x)'

    def __call__(self, input: np.array):
        return np.cos(self.n * np.pi * input)

class FeatureVector:
    """
    Class to store multiple features. When forwarded an input array returns a stack of the features
    evaluated at the given positions.
    """
    def __init__(self, features: List[Feature]):
        self.features = features

    def add_feature(self, feature: Feature):
        self.features.append(feature)

    def remove_feature(self, idx: int):
        self.features.pop(idx)

    def __call__(self, input: np.array):
        """
        Forwards the input to all features and returns the stacked outputs.
        """
        out_features = []

        for feature in self.features:
            out_features.append(feature(input))

        return np.stack(out_features).T

    def __len__(self):
        return len(self.features)
