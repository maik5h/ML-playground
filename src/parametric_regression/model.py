from typing import Callable
from dataclasses import dataclass

import scipy as sc
import numpy as np
from numpy.typing import NDArray

from ..math_utils import Feature
from ..math_utils import FeatureVector
from ..math_utils import get_gaussian
from ..math_utils import Gaussian
from ..math_utils import TrainableModel


@dataclass
class StateInfo:
    """
    Class to store information about changes in the state of a
    ParametricGaussian model.

    Attributes
    ----------

    update_plot: `bool`
        Indicates whether the density plot needs to be updated, as a
        consequence of changes in the weight distribution or feature
        vector.
    update_labels: `bool`
        Indicates whether the x- and y-label of the weight space plot
        must be updated after the displayed weights or a feature
        parameter has been changed.
    rm_feature: `int`
        The index of a removed feature. `None` if no feature has been
        removed.
    """
    update_plot: bool = False
    update_labels: bool = False
    rm_feature: int = None


class ParametricGaussian(Gaussian, TrainableModel):
    """
    Extends the Gaussian class by adding a feature vector `phi`, which
    is used to build a parametric function by multiplying it to the
    random variables of this Gaussian.
    Further adds methods to condition on data and update the
    distribution manually.
    """
    def __init__(self, phi: FeatureVector):
        """
        Initializes the Gaussian with mu = 0 and sigma = unit matching
        the dimension of the input feature vector.
        """
        if len(phi) < 2:
            raise ValueError('Feature vector must have at least two elements.')

        super().__init__(np.zeros(len(phi)), np.eye(len(phi)))
        # Phi is a feature vector defining the basis functions of the
        # model.
        self.phi: FeatureVector = phi

        # Plotting weight and function space distributions are handled
        # inside different classes.
        # The following functions are used to notify these classes
        # about changes of this instance. This is supposed to be some
        # kind of lightweight observer pattern.
        self.notify_weight_gui: Callable[[StateInfo], None] = None
        self.notify_func_gui: Callable[[StateInfo], None] = None

    def condition(self, x: NDArray, y: NDArray, sigma: NDArray) -> None:
        """
        Updates this Gaussian to represent the conditional probability
        given a linear transformation phi, data Y and the noise amount
        on the data sigma.
        """
        # TODO the data noise sigma is required here. I dont like that as in reality the noise might be unknown.
        # Is there a way to avoid it?

        phi = self.phi(x).T

        # If the data Y conatins only one datum, use simplified form of inference.
        if y.shape == (1,):
            self.mu += ((self.sigma @ phi / (phi.T @ self.sigma @ phi + sigma ** 2)) * (y - phi.T @ self.mu)).squeeze()
            self.sigma -= (self.sigma @ phi / (phi.T @ self.sigma @ phi + sigma ** 2)) @ (phi.T @ self.sigma)

        # If data has multiple points, use cholensky decomposition of the matrix
        # A = phi.T @ self.sigma @ phi + sigma ** 2 and solve linear equation instead of
        # explicitly calculating A^-1.
        else:
            fac = sc.linalg.cho_factor(phi.T @ self.sigma @ phi + sigma ** 2)

            self.mu += self.sigma @ phi @ sc.linalg.cho_solve(fac, (y - phi.T @ self.mu))
            self.sigma -= self.sigma @ phi @ sc.linalg.cho_solve(fac, phi.T @ self.sigma)
        
        self._notify_gui(StateInfo(update_plot=True))

    def add_feature(self, feature: Feature) -> None:
        """
        Adds a random variable to this Gaussian which corresponds to
        the input Feature. The plots are updated accordingly.
        """
        # Add dimension to parent Gaussian.
        self.add_random_variable()

        # Add the feature to phi and update the array of features evaluated at x-values.
        self.phi.add_feature(feature)

        self._notify_gui(StateInfo(update_plot=True))
    
    def remove_feature(self, rm_idx: int) -> None:
        """
        Removes a feature from the FeatureVector associated with this
        instance and removes the corresponding weight from this
        distribution.

        Attributes
        ----------
        rm_idx: `int`
            The index of the feature and weight to be removed.
        """
        # Remove dimension from parent Gaussian.
        self.remove_random_variable(rm_idx)

        # Remove the feature from phi and update the array of features evaluated at x-values.
        self.phi.remove_feature(rm_idx)

        # Notify the GUI objects associated with this instance about
        # the removed feature.
        self._notify_gui(StateInfo(update_plot=True, rm_feature=rm_idx))

    def update_feature_parameter(self) -> None:
        """
        Updates the function space distribution and weight space labels after
        a feature parameter has changed.
        """
        self._notify_gui(StateInfo(update_plot=True, update_labels=True))

    def set_mean(self, indices: list[int], means: list[float]) -> None:
        """
        Sets the means of the variables at the given indices to the given
        values.
        """
        for idx, mean in zip(indices, means):
            self.mu[idx] = mean

        self._notify_gui(StateInfo(update_plot=True))

    def scale_sigma(self, factor: float, indices: list[int]) -> None:
        """
        Scales the sigma matrix entries concerning the random variables
        at the given indices.

        Attributes
        ----------

        factor: `float`
            The scaling factor applied to the affected entries.
        indices: `list[int]`
            List of the indices that are supposed to be scaled.
        """
        scale = np.eye(len(self.phi))
        scale[indices, indices] = factor
        self.sigma = scale @ self.sigma @ scale.T

        self._notify_gui(StateInfo(update_plot=True))

    def rotate_sigma(self, angle: float, indices: tuple[float, float]) -> None:
        """
        Rotates the pdf of the random variables at indices around their mean.
        """
        if not len(indices) == 2:
            raise ValueError('Rotation is only supported for two indices at once.')

        # Create unity matrix and insert rotation matrix at concerned indices.
        rot = np.eye(len(self.phi))
        c, s = np.cos(angle), np.sin(angle)
        i, j = indices
        rot[[i, i, j, j], [i, j, i, j]] = [c, -s, s, c]

        self.sigma = rot @ self.sigma @ rot.T

        self._notify_gui(StateInfo(update_plot=True))

    def reset_active_variables(self, indices: list[int]) -> None:
        """
        Resets the currently displayed random variables to mean zero and diagonal covariance one.
        """
        self.mu[indices] = 0

        # TODO there must be a faster alternative without iterating.
        for i in indices:
            for j in indices:
                self.sigma[i, j] = 1 if i == j else 0

        self._notify_gui(StateInfo(update_plot=True))
    
    def get_likelihood(self, x_data: NDArray, y_data: NDArray) -> NDArray:
        """
        Calculate the probability density at pairs of `x` and `y` values.

        Parameters
        ----------
        x_data: `NDArray`
            Array of x-values with shape (N,) at which to evaluate the
            probability density.
        y_data: `NDArray`
            Array of y-values with shape (N,) at which to evaluate the
            probability density.

        Returns
        -------
        `NDArray`of shape (N,) containing probabilities to find
        function values `y` at positions `x`.
        """
        features = self.phi(x_data)

        # To obtain the distribution over function values at a given x,
        # the weights are multiplied with the feature vector phi
        # evaluated at that x:
        # phi(x) @ w = phi_0(x) * w_0 + phi_1(x) * w_1 + ...

        # The product of _features with mu creates an array of size
        # len(_features) with the new mean for every x value
        # corresponding to the entries in _features.
        mu = features @ self.mu

        # The variance of sigma transformed by a single feature is
        # given by new_sigma = feature @ sigma @ feature.T. However,
        # since broadcasting is used I deviate from this form and use
        # an element wise multiplication followed by a summation
        # instead. This performs the multiplication and subsequent
        # summation otherwise done by the inner product for every of
        # the stacked features.
        sigma = np.sum(features.T * (self.sigma @ features.T), axis=0)

        return get_gaussian(y_data, mu, sigma)
    
    def get_function_space_density(self, x_data: NDArray, y_data: NDArray) -> NDArray:
        """
        Calculate the probability density at the grid given by `x` and `y`.

        Parameters
        ----------
        x_data: `NDArray`
            Array of x-values with shape (M,) at which to evaluate the
            probability density.
        y_data: `NDArray`
            Array of y-values with shape (N,) at which to evaluate the
            probability density.

        Returns
        -------
        `NDArray` of shape (N, M) containing probabilities to find
        function values `y` at positions `x`.
        """
        # See get_likelihood for details. Only difference here is a
        # transpose in the last statement to get a 2D x-y-grid rather
        # than a 1D list of x-y-pairs.
        features = self.phi(x_data)
        mu = features @ self.mu
        sigma = np.sum(features.T * (self.sigma @ features.T), axis=0)

        return get_gaussian(y_data[None, :].T, mu, sigma)

    def _notify_gui(self, info: StateInfo) -> None:
        """
        Notify the WeightSpaceGUI and FunctionSpacePlot instances of
        changes in the model state that require updates of the plots.
        """
        self.notify_weight_gui(info)
        self.notify_func_gui(info)
