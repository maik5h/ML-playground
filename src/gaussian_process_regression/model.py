from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve

from ..math_utils import GaussianProcess, TrainableModel, get_gaussian
from ..math_utils import Kernel, RBFKernel, PolynomialKernel


class ConditionalGaussianProcess(TrainableModel):
    """Gaussian process which can be conditioned on data.

    The posterior Gaussian process is accessible through the
    `posterior` property.
    """
    def __init__(self,
                 prior_mean: Callable[[NDArray], NDArray] = lambda x: np.zeros_like(x),
                 prior_kernel: Kernel = RBFKernel()):
        """Parameters
        ----------

        prior_mean: `Callable[[NDArray], NDArray]
            Mean function of the prior Gaussian process. Defaults to 0.
        prior_kernel: `Kernel` 
            Kernel function of the prior Gaussian process. Defaults to
            an RBF kernel.
        """
        self._prior_gp = GaussianProcess(prior_mean, prior_kernel)
        self._posterior_gp = GaussianProcess(self._mu, self._kernel)
        self._x_samples = np.array([])
        self._y_samples = np.array([])

        # Helpful factors to calculate the posterior GP.
        self._covariance_lu_factor = None
        self._representer_weights = None

        # List of functions to call every time the model state changes.
        self._observer_calls: list[Callable[[], None]] = []

    def restart_training(self):
        """Clear the arrays of target samples.
        """
        self._x_samples = np.array([])
        self._y_samples = np.array([])

    @property
    def posterior(self) -> GaussianProcess:
        # Return the prior if it has not been conditioned on data.
        # Prevents errors in self._mean and self._covariance when
        # called while self._x_samples is empty.
        if len(self._x_samples) == 0:
            return self._prior_gp
        else:
            return self._posterior_gp

    def _mu(self, x: NDArray) -> NDArray:
        """Calculate the posterior mean of function values at x.

        The new mean `m'` is the product of prior kernel `k` and
        representer weights `alpha` added to the prior mean `m`:

        `m'(x) = m(x) + k(x, X) @ alpha`

        Uppercase X denotes the observed x-samples.
        """
        return (
            self._prior_gp.get_mean(x)
            + self._prior_gp._kernel(x, self._x_samples)
            @ self._representer_weights
        )

    def _kernel(self, a: NDArray, b: NDArray) -> NDArray:
        """Evaluate the posterior kernel function at a and b.

        The posterior kernel function `k'` is given as:

        `k'(a, b) = k(a, b) - k(a, X) @ k(X, X)^-1 @ k(X, b)`

        with the prior kernel function `k` and observed x-values `X`.
        """
        return (
            self._prior_gp._kernel(a, b)
            - self._prior_gp._kernel(a, self._x_samples)
            @ lu_solve(self._covariance_lu_factor, self._prior_gp._kernel(self._x_samples, b))
        )

    def condition(self, x: NDArray, y: NDArray, _sigma: float) -> None:
        # Every time `condition` is called, the posterior is calculated
        # from scratch, ignoring the previous state. This is convenient
        # and the alternative would involve many nested function calls.
        # Conditioning has cubic time complexity (involves LU
        # decomposition) but should be fast enough as long as the
        # number of target samples is small.
        self._x_samples = np.concatenate((self._x_samples, x))
        self._y_samples = np.concatenate((self._y_samples, y))

        # LU factorization of the matrix k_XX + sigma.
        self._covariance_lu_factor = lu_factor(
            self._prior_gp._kernel(
                self._x_samples,
                self._x_samples
            )
        )

        # Representer weights (k_XX + sigma)^-1 @ (y - m_X).
        self._representer_weights = lu_solve(
            self._covariance_lu_factor,
            self._y_samples - self._prior_gp.get_mean(self._x_samples)
        )

        for func in self._observer_calls:
            func()

        # The training must be stopped if the kernel is polynomial and
        # the number of samples exceeds the number of dimensions.
        if isinstance(self._prior_gp._kernel, PolynomialKernel):
            if len(self._x_samples) == self._prior_gp._kernel._power:
                return False

        return True

    def get_likelihood(self, x: NDArray, y: NDArray) -> NDArray:
        mu = self._posterior_gp.get_mean(x)
        sigma = self._posterior_gp.get_sigma(x)
        likelihood = get_gaussian(y, mu, sigma)
        return likelihood

    def add_observer_call(self, func: Callable[[], None]) -> None:
        """Add a function to be called every time the model state changes.
        """
        self._observer_calls.append(func)
