import abc
from typing import Union, Callable

import numpy as np
from numpy.typing import NDArray


# Kernels are defined by their ability to be called on two arrays `a`
# and `b` and return a positive semidefinite matrix.
# The kernel function `f` is evaluated at pairs of the input
# values and returned as a matrix `k` where `k[i, j] =
# f(a[i], b[j])`.
Kernel = Callable[[NDArray, NDArray], NDArray]


class KernelInterface(metaclass=abc.ABCMeta):
    """Abstract base class to store and access additional parameters
    of kernel functions.
    """
    @property
    @abc.abstractmethod
    def parameters(self) -> list[Union[int, float]]:
        """Returns the parameters of this kernel as a list.
        """
        pass

    @abc.abstractmethod
    def set_parameter(self, idx: int, value: Union[int, float]) -> None:
        """Sets the parameter at index `idx` to `value`.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_parameter_limits() -> list[tuple[float, float, float]]:
        """Returns a list of the minimum value, maximum value and intended
        step size for ever parameter of this type of kernel.
        """
        pass


class RBFKernel(Kernel, KernelInterface):
    def __init__(self, in_scale: float = 1, out_scale: float = 2.5):
        """Creates a Radial basis function (RBF) kernel with `in_scale`
        and `out_scale`.

        Parameters
        ----------

        in_scale: `float`
            Defines the scale on which the covariance between function
            values decays. High values produce smooth curves, low
            values fluctuating curves.
        out_scale: `float`
            Scales the output of the kernel. This value is equal to the
            standard deviation of the resulting function space
            distribution.
        """
        self._in_scale = in_scale
        self._out_scale = out_scale

    def __call__(self, a: NDArray, b: NDArray) -> NDArray:
        """Returns the kernel function evaluated at (a, b).

        The kernel function is defined as
        `out_scale^2 * exp((a.T - b) / (2 * in_scale^2))`.
        """
        exponent = -(((a[:, None] - b) / self._in_scale) ** 2) / 2
        return self._out_scale ** 2 * np.exp(exponent)

    @property
    def parameters(self) -> list[float]:
        """Returns [in_scale, out_scale].
        """
        return [self._in_scale, self._out_scale]

    def set_parameter(self, idx: int, value: float) -> None:
        """Sets:
        - idx == 0: in_scale
        - idx == 1: out_scale
        """
        if idx == 0:
            self._in_scale = value
        elif idx == 1:
            self._out_scale = value
        else:
            raise ValueError(f'Tried to set parameter number {idx}, while only two are available.')

    @staticmethod
    def get_parameter_limits() -> list[tuple[float, float, float]]:
        """Returns the minimum value, maximum value and intended step
        size for `in_scale` and `out_scale`:

        in_scale:   min=0.05,   max=1,  step=0.05
        out_scale:  min=0.05,   max=1,  step=0.05
        """
        in_lims = (0.05, 1, 0.05)
        out_lims = (0.05, 1, 0.05)
        return [in_lims, out_lims]


class PolynomialKernel(Kernel, KernelInterface):
    def __init__(self, power: int = 1, offset: float = 1):
        """Create a polynomial kernel with `power` and `offset`.

        Gaussian processes with this kernel produce polynomial
        functions of degree `power`.

        Parameters
        ----------

        power: `int`
            Degree of the polynomial. Must be an integer >= 0.
        offset: `float`
            Offset added to the product of inputs before
            exponentiating. Offsets > 0 lead to non-zero variance
            at x=0.
        """
        if power < 0:
            raise ValueError('Power must be non-negative.')
        if offset < 0:
            raise ValueError('Offset must be non-negative.')
        self._power = power
        self._offset = offset

    def __call__(self, a: NDArray, b: NDArray) -> NDArray:
        """Returns the kernel function evaluated at (a, b).

        The kernel function is given as
        `(a.T * b + offset)^power`.
        """
        return (a[:, None] * b + self._offset) ** self._power

    @property
    def parameters(self) -> list[float]:
        """Returns [power, offset].
        """
        return [self._power, self._offset]

    def set_parameter(self, idx: int, value: float) -> None:
        """Sets:
        - idx == 0: power
        - idx == 1: offset
        """
        if idx == 0:
            self._power = value
        elif idx == 1:
            self._offset = value
        else:
            raise ValueError(f'Tried to set parameter number {idx}, while only two are available.')

    @staticmethod
    def get_parameter_limits() -> list[tuple[float, float, float]]:
        """Returns the minimum value, maximum value and intended step
        size for `power` and `offset`:

        power:    min=1,    max=10, step=1
        offset:   min=0,    max=1,  step=0.1
        """
        power_lims = (1, 4, 1)
        offset_lims = (0, 1, 0.1)
        return [power_lims, offset_lims]


class KernelProduct(Kernel):
    """Class to store multiple kernels and compute their product.
    """
    def __init__(self, kernels: set[Kernel]):
        self._kernels = kernels

    def __call__(self, a: NDArray, b: NDArray) -> NDArray:
        """Return the element wise product of kernel matrices.
        """
        out = 1
        for kernel in self._kernels:
            out *= kernel(a, b)

        return out

    def add_kernel(self, kernel: Kernel) -> None:
        """Multiply a new kernel to the kernel product.
        """
        self._kernels.add(kernel)

    def remove_kernel(self, kernel: Kernel) -> None:
        """Remove the given kernel from the product.
        """
        self._kernels.remove(kernel)


class KernelSum(Kernel):
    def __init__(self, kernels: set[Kernel]):
        self._kernels = kernels

    def __call__(self, a: NDArray, b: NDArray) -> NDArray:
        out = 0
        for kernel in self._kernels:
            out += kernel(a, b)

        return out

    def add_kernel(self, kernel: Kernel) -> None:
        """Add a new kernel to the kernel sum.
        """
        self._kernels.add(kernel)

    def remove_kernel(self, kernel: Kernel) -> None:
        """Remove the given kernel from the sum.
        """
        self._kernels.remove(kernel)
