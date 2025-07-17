import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from gaussians import Gaussian, get_gaussian

def plot_gaussian(gaussian: Gaussian):
    mean = gaussian.mu
    cov = gaussian.sigma
    x1, x2 = np.meshgrid(
        np.linspace(0, 10, 100),
        np.linspace(0, 10, 100)
    )

    pos = np.dstack((x1, x2))
    rv = multivariate_normal(mean, cov)
    plt.contourf(x1, x2, rv.pdf(pos), levels=100, cmap='Blues')
    plt.colorbar()
    plt.title('Gaussian Distribution')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def plot_function_distribution(ax: plt.axis, gaussian: Gaussian, x: np.array, y: np.array, phi):
    """
    gaussian:   A multivariant Gaussian distribution.
    x:          Array of equidistant x-values at which to evaluate the density.
    y:          Array of equidistant y-value at which to evaluate the density.
    phi:        The feature vector
    """
    # Project into function space.
    function_dist = gaussian.project(phi)
    mu = function_dist.mu
    sigma = function_dist.sigma.diagonal()

    y_density = np.linspace(y[0], y[-1])

    # TODO use x and phi to get actual x array

    densities = []
    for i in range(x.size):
        column = get_gaussian(y, mu[i], sigma[i])
        densities.append(column)

    densities = np.stack(densities).T
    ax.imshow(densities, cmap='Blues', aspect='auto', extent=(x[0], x[-1], y[0], y[-1]))


if __name__ == "__main__":
    mean = np.array([5, 5])
    sigma = np.array([[1, 3], [1, 2]])
    gaussian = Gaussian(mean, sigma)
    plot_gaussian(gaussian)

    gaussian.project(np.array([[1, 0], [0, 0.5]]))  # Example projection
    plot_gaussian(gaussian)