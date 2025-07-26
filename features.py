import numpy as np
from typing import SupportsIndex


class Feature:
    """
    Base class for Features.
    """
    def __init__(self, parameter: int):
        self.parameter = parameter
    
    def get_expression(self) -> str:
        """
        Returns the mathematical expression of this feature as a string, for example: 'x^2'.
        """
        pass

class PolynomialFeature(Feature):
    """
    Polynomial feature with an integer power. The 'parameter' attribute of the Feature base class
    is interpreted as the power.
    """
    def __init__(self, power: int):
        super().__init__(power)

    def __call__(self, input: np.array) -> np.array:
        return input ** self.parameter
    
    def get_expression(self) -> str:
        if self.parameter == 0:
            name = '1'
        elif self.parameter == 1:
            name = 'x'
        else:
            name = f'x^{self.parameter}'

        return name

class SineFeature(Feature):
    """
    Sine feature with frequency of a natural multiple of pi. The 'parameter' attribute of the
    Feature base class is interpreted as half of the frequency.
    """
    def __init__(self, n_pi: int):
        super().__init__(n_pi)

    def __call__(self, input: np.array) -> np.array:
        return np.sin(self.parameter * np.pi * input)
    
    def get_expression(self) -> str:
        if self.parameter == 1:
            name = '\sin(\pi x)'
        else:
            name = f'\sin({self.parameter}\pi x)'

        return name

class CosineFeature(Feature):
    """
    Cosine feature with frequency of a natural multiple of pi. The 'parameter' attribute of
    the Feature base class is interpreted as half of the frequency.
    """
    def __init__(self, n_pi: int):
        super().__init__(n_pi)

    def __call__(self, input: np.array) -> np.array:
        return np.cos(self.parameter * np.pi * input)
    
    def get_expression(self) -> str:
        if self.parameter == 1:
            name = '\cos(\pi x)'
        else:
            name = f'\cos({self.parameter}\pi x)'

        return name

class FeatureVector:
    """
    Class to store multiple features. When forwarded an input array of x-values, returns a stack of the features
    evaluated at the given positions [phi_1(x), phi_2(x), ...].T.
    """
    def __init__(self, features: list[Feature]):
        self.features = features

    def add_feature(self, feature: Feature) -> None:
        self.features.append(feature)

    def remove_feature(self, idx: int) -> None:
        self.features.pop(idx)

    def __call__(self, input: np.array) -> np.array:
        """
        Forwards the input to all features and returns the stacked outputs [phi_1(x), phi_2(x), ...].T.
        """
        out_features = []

        for feature in self.features:
            out_features.append(feature(input))

        return np.stack(out_features).T

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index: SupportsIndex) -> Feature:
        return self.features[index]
    