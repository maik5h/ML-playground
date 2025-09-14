import numpy as np
from numpy.typing import NDArray

from . import FeatureVector
from . import PolynomialFeature
from . import HarmonicFeature
from . import GaussFeature


# Tuple of all Feature classes.
available_features = (PolynomialFeature, HarmonicFeature, GaussFeature)

class FeatureSampleGenerator:
    """Basic generator for random pairs of x- and y-samples.

    Calling creates a random function built from one or two of the
    features defined in src/math_utils/features.py. Gaussian noise is
    added and random samples are drawn.
    """
    def __init__(self,
                 n_samples: int,
                 noise_amount: float,
                 xlim: tuple[float, float],
                 max_weight: float = 3.2):
        """Parameters
        ----------
        
        n_samples: `int`
            The number of samples returned on every call.
        noise_amount: `float`
            The amount of Gaussian noise to be added to the samples.
        xlim: `(float, float)`
            The x-range to sample from.
        max_weight: `float`
            The maximum coefficient a feature can be multiplied with.
            Coefficients are drawn from a uniform distribution over
            [0, max_weight).
        """
        self._n_samples = n_samples
        self._noise_amount = noise_amount
        self._xlim = xlim
        self._max_weight = max_weight
    
    def __call__(self) -> tuple[NDArray, NDArray]:
        """
        Generates a random set of x-y-pairs.
        """
        # Create a random target function using a FeatureVector.
        target_function = FeatureVector([])
        n_features = np.random.randint(1, 3)
        for _ in range(n_features):
            feature_type = np.random.randint(0, len(available_features))

            # Retrieve list of parameters allowed for the chosen type
            # of feature and select random elements.
            parameters = []
            for lims in available_features[feature_type].get_parameter_limits():
                # Determine valid parameters and choose a random value.
                valid_ps = np.arange(lims[0], lims[1], lims[2])
                parameters.append(valid_ps[np.random.randint(len(valid_ps))])

            target_function.add_feature(available_features[feature_type](*parameters))

        # Evaluate the target function at n_samples x positions. This
        # returns an array with each feature evaluated at x separately
        # with shape=(n_samples, n_features).
        x_samples = np.linspace(self._xlim[0], self._xlim[1], self._n_samples)
        y_features = target_function(x_samples)

        # Create a weight for each feature and make sure all weights
        # lie well inside the plotted space. Multiply them to the
        # feature array.
        weights = np.random.rand(n_features) * self._max_weight * 2 - self._max_weight
        y_samples = y_features @ weights

        y_samples += np.random.normal(0, scale=self._noise_amount, size=y_samples.shape)

        return x_samples, y_samples
