from typing import Union
from logging import warning

import numpy as np
from numpy.typing import NDArray


def get_gaussian(x: NDArray, mu: NDArray, sigma: NDArray) -> Union[float, NDArray]:
    """
    Returns the value of a Gaussian function with mean mu and standard deviation sigma at value x.
    """
    coeff = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * np.exp(exponent)

class Gaussian:
    """
    Class representing multivariate Gaussian distributions.
    The probability density is Gaussian and characterized by the mean
    vector `mu` and the covariance matrix `sigma`.
    """
    def __init__(self, mu: NDArray, sigma: NDArray):
        self.mu = mu
        self.sigma = sigma
    
    def project(self, A: NDArray):
        """
        Applies the projection A to this distribution and returns a new Gaussian with corresponding mu and sigma.
        """
        mu = A @ self.mu
        sigma = A @ self.sigma @ A.T

        return Gaussian(mu, sigma)
    
    def add_random_variable(self) -> None:
        """
        Adds a new random variable to this Gaussian. The new variable has a mean of zero and covariance of one.
        """
        self.mu = np.append(self.mu, 0)

        previous_sigma = self.sigma
        self.sigma = np.eye(len(self.mu))

        self.sigma[:-1, :-1] = previous_sigma
    
    def remove_random_variable(self, idx: int) -> None:
        """
        Removes the random variable at index idx from the distribution resulting in the joint distribution
        of the remaining variables.
        """
        if self.sigma.size == 2:
            warning("Tried to reduce the dimension of a Gaussian with two variables, which is not supported.")
            return
        
        # Gaussian distributions are closed under marginalization, so removing a random variable is
        # equivalent to removing the correpsonding element from mu and the corresponding rows and
        # columns from sigma.
        self.mu = np.delete(self.mu, idx)
        self.sigma = np.delete(self.sigma, idx, axis=0)
        self.sigma = np.delete(self.sigma, idx, axis=1)
    
    def select_random_variables(self, indices: tuple[int, int]) -> tuple[NDArray, NDArray]:
        """
        Selects the random variables at the input indices by marginalizing out all other variables.
        Returns the corresponding mean vector mu and covariance matrix sigma.

        Only supports the selection of exactly two variables.
        """
        # When considering the order of indices, it is more straightforward to just implement the relevant
        # case of len(indices) == 2 instead of a general method. (For now, this may be subject to future efforts)
        if not len(indices) == 2:
            raise ValueError("Selecting any number of random variables other than two is not supported.")

        mu = self.mu
        sigma = self.sigma

        # Find the indices to remove.
        rm_indices = [len(self.mu) - n - 1 for n in range(len(self.mu))]
        for i in indices:
            rm_indices.remove(i)

        for idx in rm_indices:
            mu = np.delete(mu, idx)
            sigma = np.delete(sigma, idx, axis=0)
            sigma = np.delete(sigma, idx, axis=1)
        
        # If indices is in descending order, flip the resulting mu and sigma arrays.
        if indices[0] > indices[1]:
            mu[0], mu[1] = mu[1], mu[0]
            sigma = np.flip(sigma, (0, 1))

        return mu, sigma
