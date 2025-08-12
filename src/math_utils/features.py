import abc
from typing import SupportsIndex, Union

import numpy as np
from scipy.special import factorial
from numpy.typing import NDArray


class Feature(metaclass=abc.ABCMeta):
    """
    Base class for features, i.e. 1D functions that are used to build
    parametric models. All features are parameterized by at least one
    value. The interpretation of these values is subject of the
    implementations of the subclasses, they may for example denote
    frequency and phase of a harmonic function.
    """
    @property
    @abc.abstractmethod
    def parameters(self) -> list[Union[int, float]]:
        """
        Returns the parameters of this feature as a list.
        """
        pass
    
    @abc.abstractmethod
    def set_parameter(self, idx: int, value: Union[int, float]) -> None:
        """
        Sets the parameter at index `idx` to `value`.
        """
        pass
    
    @abc.abstractmethod
    def get_expression(self) -> str:
        """
        Returns the mathematical expression of this feature as a
        string, for example: 'x^2'.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_parameter_limits() -> list[tuple[float, float, float]]:
        """
        Returns a list of the minimum value, maximum value and intended
        step size for ever parameter of this feature.
        """
        pass


class PolynomialFeature(Feature):
    """
    Polynomial feature with an integer power. The exponent is the only
    parameter of this Feature.
    """
    def __init__(self, power: int):
        self._power = power

    @property
    def parameters(self) -> list[int]:
        return [self._power,]
    
    def set_parameter(self, idx: int, value: int) -> None:
        if idx == 0:
            self._power = value
        else:
            raise ValueError(f'Trying to set parameter {idx} of PolynomialFeature, which has one parameter.')

    def __call__(self, input: NDArray) -> NDArray:
        # Scale the output with the inverse factorial of the power to prevent
        # higher order terms to overwhelm lower order terms.
        return input ** self._power / factorial(self._power)

    def get_expression(self) -> str:
        if self._power == 0:
            name = '1'
        elif self._power == 1:
            name = 'x'
        else:
            name = f'x^{self._power}'

        return name
    
    @staticmethod
    def get_parameter_limits() -> list[tuple[float, float, float]]:
        """
        Returns the minimum value, maximum value and intended step size
        of the exponent:
            min=0,
            max=4,
            step_size=1
        """
        return [(0, 4, 1),]


class HarmonicFeature(Feature):
    """
    Harmonic feature with frequency of a natural multiple of pi. The
    parameter at index 0 denotes half of the frequency, the parameter
    at index 1 is the phase.
    """
    def __init__(self, frequency: int, phase: int):
        self._frequency = frequency
        self._phase = phase

    @property
    def parameters(self) -> list[int]:
        return [self._frequency, self._phase]
    
    def set_parameter(self, idx: int, value: list[int]) -> None:
        if idx == 0:
            self._frequency = value
        elif idx == 1:
            self._phase = value
        else:
            raise ValueError(f'Trying to set parameter {idx} of HarmonicFeature, which has two parameters.')

    def __call__(self, input: NDArray) -> NDArray:
        omega = self._frequency  * np.pi
        phase = self._phase / 2 * np.pi
        return np.sin(omega * input - phase)
    
    def get_expression(self) -> str:
        # Determine the sign of the phase and create tuple of possible phases.
        sign = '+' if self._phase > 0 else '-'
        phase_strings = ('', f' {sign} \pi/2', f' {sign} \pi', f' {sign} 3\pi/2')

        # Select the correct string corresponding to the phase.
        phase_string = phase_strings[abs(self._phase) % 4]

        # Add the frequency if needed.
        if self._frequency == 1:
            name = f'\sin(\pi x{phase_string})'
        else:
            name = f'\sin({self._frequency}\pi x{phase_string})'

        return name
    
    @staticmethod
    def get_parameter_limits() -> list[tuple[float, float, float]]:
        """
        Returns the minimum value, maximum value and intended step size for
        all parameters:

        frequency:  min=1,  max=4,  step_size=1
        phase:      min=0, max=3,  step_size=1
        """
        return [(1, 4, 1), (0, 3, 1)]


class GaussFeature(Feature):
    """
    Gaussian feature with frequency of a natural multiple of pi. The
    parameter at index 0 is the standard deviation of the curve,
    the parameter at index 1 is the mean.
    """
    def __init__(self, sigma: float, mu: float):
        self._sigma = sigma
        self._mu = mu

    @property
    def parameters(self) -> list[float]:
        return [self._sigma, self._mu]

    def set_parameter(self, idx: int, value: float) -> None:
        if idx == 0:
            self._sigma = value
        elif idx == 1:
            self._mu = value
        else:
            raise ValueError(f'Trying to set parameter {idx} of GaussFeature, which has two parameters.')

    def __call__(self, input: NDArray) -> NDArray:
        coeff = 1 / (np.sqrt(2 * np.pi) * self._sigma)
        exponent = -0.5 * ((input - self._mu) / self._sigma) ** 2
        return coeff * np.exp(exponent)

    def get_expression(self) -> str:
        return r'\varphi_{' + f'\mu={self._mu:.1f}, \sigma={self._sigma:.1f}' + '}(x)'

    @staticmethod
    def get_parameter_limits() -> list[tuple[float, float, float]]:
        """
        Returns the minimum value, maximum value and intended step size
        for all parameters:

        sigma:  min=0.2,    max=1,  step_size=0.2
        mu:     min=-4,     max=4,  step_size=0.1
        """
        return [(0.2, 1, 0.2), (-4, 4, 0.1)]


class FeatureVector:
    """
    Class to store multiple features. When forwarded an input array of x-values, returns a stack of the features
    evaluated at the given positions [phi_1(x), phi_2(x), ...].T.
    """
    def __init__(self, features: list[Feature] = []):
        self.features = features

    def add_feature(self, feature: Feature) -> None:
        self.features.append(feature)

    def remove_feature(self, idx: int) -> None:
        self.features.pop(idx)

    def __call__(self, input: NDArray) -> NDArray:
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
