from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve

from ..math_utils import GaussianProcess, TrainableModel, get_gaussian
from ..math_utils import Kernel, RBFKernel, PolynomialKernel, KernelSum, KernelProduct


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

    def _get_max_training_steps(self) -> Optional[int]:
        """Get the maximum number of training steps allowed by the kernel.

        If the model is trained with a polynomial kernel, the number of
        datapoints it is conditioned on is not allowed to exceed the
        exponent of the kernel function. This method returns the
        maximum number of datapoints or `None`, if there is no limit.
        """
        # TODO I do not like the fact i have to do this. There must be
        # a cleaner way.

        k = self._prior_gp._kernel
        # Check if the kernel is a polynomial kernel.
        if isinstance(k, PolynomialKernel):
                return k._power + 1
        # Check if the kernel is a KernelSum or KernelProduct and the
        # sum/ product has only one polynomial kernel as a term.
        elif isinstance(k, (KernelSum, KernelProduct)):
            if len(k) == 1:
                k = list(k._kernels)[0]
                if isinstance(k, PolynomialKernel):
                    return k._power + 1


    def condition(self, x: NDArray, y: NDArray, _sigma: float) -> None:
        """Condition the Gaussian process on data.

        If `x` or `y` is `None`, the posterior is recalculated on the
        same data the model was conditioned on before.
        """
        # Every time `condition` is called, the posterior is calculated
        # from scratch, ignoring the previous state. This is allows
        # changing the prior kernel and receiving immediate posterior
        # updates at the current stage of training.
        # Conditioning has cubic time complexity (involves LU
        # decomposition) but should be fast enough as long as the
        # number of target samples is small.
        if x is not None and y is not None:
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

        # Stop the training if the maximum allowed number of training
        # steps has been reached.
        lim = self._get_max_training_steps()
        if lim is not None and len(self._x_samples) >= lim:
            return False

        return True

    def get_likelihood(self, x: NDArray, y: NDArray) -> NDArray:
        """Return the probability to find a sample `y` at `x`.
        """
        gp = self._prior_gp if len(self._x_samples) == 0 else self._posterior_gp
        mu = gp.get_mean(x)
        sigma = gp.get_sigma(x)
        likelihood = get_gaussian(y, mu, sigma)
        return likelihood

    def add_observer_call(self, func: Callable[[], None]) -> None:
        """Add a function to be called every time the model state changes.
        """
        self._observer_calls.append(func)

    def refresh(self) -> None:
        """Recalculate the posterior without changing the loaded data.

        This can be called when the prior kernel is changed to update
        the posterior. Consequently, kernel parameters can seemingly be
        changed while preserving the conditioned state of the previous
        model. In reality, the posterior is recalculated under the
        hood.
        """
        # If the model has been conditioned at least once, condition it
        # again on the same data.
        if len(self._x_samples) > 0:
            # Truncate the data if the new kernel allows for less
            # training steps than previously performed.
            lim = self._get_max_training_steps()
            if lim is not None and len(self._x_samples) > lim:
                self._x_samples = self._x_samples[:lim]
                self._y_samples = self._y_samples[:lim]

            self.condition(None, None, None)
        # If the model has not been conditioned yet, just inform the
        # observer of the changes.
        else:
            for func in self._observer_calls:
                func()
