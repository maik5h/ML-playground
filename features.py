import numpy as np
from scipy.special import factorial
from typing import SupportsIndex
import abc


class Feature(metaclass=abc.ABCMeta):
    """
    Base class for features. All features are parameterized by two values, parameter_a and parameter_b.
    The interpretation of these values is subject of the implementations of the subclasses, they may
    for example denote exponent and x-offset of a polynomial term.
    """
    @property
    @abc.abstractmethod
    def parameter_a(self):
        """
        Primary parameter, for example exponent or frequency.
        """
        pass

    @property
    @abc.abstractmethod
    def parameter_b(self):
        """
        Secondary parameter, for example x-offset or phase.
        """
        pass
    
    @parameter_a.setter
    @abc.abstractmethod
    def parameter_a(self, value):
        pass

    @parameter_b.setter
    @abc.abstractmethod
    def parameter_b(self, value):
        pass
    
    @abc.abstractmethod
    def get_expression(self) -> str:
        """
        Returns the mathematical expression of this feature as a string, for example: 'x^2'.
        """
        pass

class PolynomialFeature(Feature):
    """
    Polynomial feature with an integer power. The 'parameter_a' attribute of the Feature base class
    is interpreted as the power, 'parameter_b' is the x-offset.
    """
    def __init__(self, power: int, offset: int):
        self._power = power
        self._offset = offset

    @property
    def parameter_a(self) -> int:
        return self._power

    @property
    def parameter_b(self) -> int:
        return self._offset
    
    @parameter_a.setter
    def parameter_a(self, value: int) -> None:
        self._power = value

    @parameter_b.setter
    def parameter_b(self, value: int) -> None:
        self._offset = value

    def __call__(self, input: np.array) -> np.array:
        # Scale the output with the inverse factorial of the power to prevent
        # higher order terms to overwhelm lower order terms.
        return (input + self._offset) ** self._power / factorial(self._power)

    def get_expression(self) -> str:
        # Turn the expression into a fancy string, i.e. only add parentheses and exponent if required
        # and just return '1' if power is zero.
        if self._power == 0:
            return '1'
        
        offset_sign = '+' if self._offset > 0 else '-'

        # Determine if offset is needed in string and if parentheses should be added.
        if self._offset == 0:
            base = 'x'
        elif self._power == 1:
            base = f'x {offset_sign} {abs(self._offset)}'
        else:
            base = f'(x {offset_sign} {abs(self._offset)})'
        
        # Add exponent if necessary.
        if self._power == 1:
            name = base
        else:
            name = f'{base}^{self._power}'

        return name

class HarmonicFeature(Feature):
    """
    Harmonic feature with frequency of a natural multiple of pi. The 'parameter_a' attribute of the
    Feature base class is interpreted as half of the frequency, 'parameter_b' is the x_offset.
    """
    def __init__(self, frequency: int, phase: int):
        self._frequency = frequency
        self._phase = phase

    @property
    def parameter_a(self) -> int:
        return self._frequency

    @property
    def parameter_b(self) -> int:
        return self._phase
    
    @parameter_a.setter
    def parameter_a(self, value: int) -> None:
        self._frequency = value

    @parameter_b.setter
    def parameter_b(self, value: int) -> None:
        self._phase = value

    def __call__(self, input: np.array) -> np.array:
        return np.sin(self._frequency * np.pi * input - self._phase / 2 * np.pi)
    
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

class GaussFeature(Feature):
    """
    Gaussian feature with frequency of a natural multiple of pi. The 'parameter_a' attribute of the
    Feature base class is interpreted as the variance of the curve, 'parameter_b' is the mean.
    """
    def __init__(self, sigma: int, mu: int):
        self._sigma = sigma
        self._mu = mu

    @property
    def parameter_a(self) -> int:
        return self._sigma

    @property
    def parameter_b(self) -> int:
        return self._mu

    @parameter_a.setter
    def parameter_a(self, value: int) -> None:
        self._sigma = value

    @parameter_b.setter
    def parameter_b(self, value: int) -> None:
        self._mu = value

    def __call__(self, input: np.array) -> np.array:
        coeff = 1 / (np.sqrt(2 * np.pi) * self._sigma / 5)
        exponent = -0.5 * ((input - self._mu) / self._sigma * 5) ** 2
        return coeff * np.exp(exponent)

    def get_expression(self) -> str:
        return r'\varphi_{' + f'\mu={self._mu}, \sigma={(self._sigma / 5):.1f}' + '}(x)'

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
    