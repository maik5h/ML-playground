import numpy as np
from scipy import linalg

def get_gaussian(x, mu, sigma):
    coeff = 1 / (np.sqrt(2 * np.pi + sigma))
    exponent = -0.5 * (x - mu) ** 2 / sigma
    return coeff * np.exp(exponent)

class Gaussian:
    def __init__(self, mu: np.array, sigma: np.array):
        self.mu = mu
        self.sigma = sigma
    
    def __call__(self, x: np.array) -> float:
        d = x.shape[0]
        coeff = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(self.sigma) ** 0.5)
        exponent = -0.5 * (x - self.mu).T @ np.linalg.inv(self.sigma) @ (x - self.mu)
        return coeff * np.exp(exponent)
    
    def project(self, A: np.array):
        mu = A @ self.mu
        sigma = A @ self.sigma @ A.T

        return Gaussian(mu, sigma)

    # def condition(self, A: np.array, y: np.array):
    #     dec = linalg.cho_factor(self.sigma @ A.T @ (A @ self.sigma @ A.T))

    #     mu = linalg.cho_solve(dec, (y - A @ self.mu))
    #     sigma = linalg.cho_solve(dec, A @ self.sigma)

    #     return Gaussian(mu, sigma)
