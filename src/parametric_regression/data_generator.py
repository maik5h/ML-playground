import numpy as np
from numpy.typing import NDArray

from ..math_utils import FeatureVector
from ..math_utils import PolynomialFeature
from ..math_utils import HarmonicFeature
from ..math_utils import GaussFeature
from ..config import Config


# Tuple of all Feature classes.
available_features = (PolynomialFeature, HarmonicFeature, GaussFeature)

def generate_target_samples(n_samples: int, noise_amount: float) -> tuple[NDArray, NDArray]:
    """
    Generates a randomized selection of samples.

    Creates a function from the Features defined in features.py with weights and parameters within
    the range used by the gaussians.InteractiveGaussian plots and feature_controls.FeatureController
    interface. This makes sure that the target funciton can always be approached using the interface.
    Gaussian noise is added to the function values.

    Returns x- and y-values of the samples in separate arrays.

    Parameters
    ----------

    n_samples: `int`
        Number of samples to take from the target function.
    noise_amount: `float`
        Variance of the Gaussian noise added to the samples.
    """
    # Create a random target function using a FeatureVector.
    target_function = FeatureVector([])
    n_features = np.random.randint(1, 3)
    for _ in range(n_features):
        feature_type = np.random.randint(0, len(available_features))

        # Choose parameters that lies within the available weight space.
        parameters = []
        for lims in available_features[feature_type].get_parameter_limits():
            # Determine valid parameters and choose a random value.
            valid_ps = np.arange(lims[0], lims[1], lims[2])
            parameters.append(valid_ps[np.random.randint(len(valid_ps))])

        target_function.add_feature(available_features[feature_type](*parameters))

    # Evaluate the target function at n_samples x positions. This returns an array with each
    # feature evaluated at x separately with shape=(n_samples, n_features).
    x_range = Config.function_space_xlim
    x_samples = np.linspace(x_range[0], x_range[1], n_samples)
    y_features = target_function(x_samples)

    # Create a weight for each feature and make sure all weights lie well inside the plotted space.
    weights = np.random.rand(n_features) * Config.weight_space_xlim[1] * 1.6 - Config.weight_space_xlim[1] * 0.8
    y_samples = y_features @ weights

    y_samples += np.random.normal(0, scale=noise_amount, size=y_samples.shape)

    return x_samples, y_samples
