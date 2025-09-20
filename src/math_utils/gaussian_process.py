from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .gaussians import Gaussian
from .kernels import Kernel, RBFKernel


class GaussianProcess:
    """Class representing a Gaussian process.

    A Gaussian process is a probability distribution over functions
    where every finite selection of function values follows a
    multivariate Gaussian distribution. It is defined by a mean
    function and kernel.
    """
    def __init__(self,
                 mu: Callable[[NDArray], NDArray] = lambda x: np.zeros_like(x),
                 kernel: Kernel = RBFKernel()):
        """Parameters
        ----------

        mu: `Callable[[NDArray], NDArray]`
            The mean function of this Gaussian process. Defaults to
            `f(x) = 0`.
        kernel: `Callable[[NDArray, NDArray], NDArray]
            The kernel function of this Gaussian Process. Must be a
            function that takes two 1D arrays as arguments and returns
            a positive semidefinite 2D array. Defaults to an RBF Kernel.
        """
        self._mu = mu
        self._kernel = kernel

    def __call__(self, x: NDArray) -> Gaussian:
        """Returns the Gaussian random variable at x.
        """
        return Gaussian(self._mu(x), self._kernel(x, x))

    def get_mean(self, x: NDArray) -> NDArray:
        """Returns the mean function evaluated at x.
        """
        return self._mu(x)

    def get_sigma(self, x: NDArray) -> NDArray:
        """Returns the kernel evaluated at (x, x), i.e. the diagonal of
        the covariance matrix.
        """
        # TODO convenient implementation, inefficient computation.
        covariance = self._kernel(x, x)
        return np.diag(covariance)
