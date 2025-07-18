import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from typing import Union


def get_gaussian(x, mu, sigma) -> Union[float, np.array]:
    """
    Returns the value of a Gaussian funktion with mean mu and standard deviation sigma at value x.
    """
    coeff = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * np.exp(exponent)

class Gaussian:
    """
    Class representing multivariate Gaussian distributions.
    The probability density is Gaussian and characterized by the mean vector mu and the covariance matrix sigma.
    """
    def __init__(self, mu: np.array, sigma: np.array):
        self.mu = mu
        self.sigma = sigma
    
    def project(self, A: np.array):
        """
        Applies the projection A to this distribution and returns a new Gaussian with corresponding mu and sigma.
        """
        mu = A @ self.mu
        sigma = A @ self.sigma @ A.T

        return Gaussian(mu, sigma)

    # def condition(self, A: np.array, y: np.array):
    #     dec = linalg.cho_factor(self.sigma @ A.T @ (A @ self.sigma @ A.T))

    #     mu = linalg.cho_solve(dec, (y - A @ self.mu))
    #     sigma = linalg.cho_solve(dec, A @ self.sigma)

    #     return Gaussian(mu, sigma)


class InteractiveGaussian(Gaussian):
    """
    Extends the Gaussian class by methods to plot the distribution and update parameters based on matplotlib callbacks.
    """
    def __init__(self, mu: np.array, sigma: np.array):
        Gaussian.__init__(self, mu, sigma)

        # Is the user currently dragging the weight distribution?
        self.dragging = False

        # The two weight distributions to display on the weight space plot.
        self.displayed_weights = [0, 1]

        # Matplotlib plots to 
        self.weight_plot = None
        self.func_plot = None

    def set_axes(self, weight_space_ax: plt.axis, function_space_ax: plt.axis) -> None:
        """
        Sets the two axes to which the weight space and function space distributions are plotted.
        Creates a weight density array, which is reused as long as sigma does not change. Initializes
        the plots.
        """
        self.ax_weight = weight_space_ax
        self.ax_func = function_space_ax

        # Freeze an array of weight space densities for faster plotting.
        xlim = self.ax_weight.set_xlim()
        ylim = self.ax_weight.set_ylim()
        self.x1, self.x2 = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 50),
            np.linspace(ylim[0], ylim[1], 50)
        )
        pos = np.dstack((self.x1, self.x2))
        self.weight_density = multivariate_normal.pdf(pos, np.zeros_like(self.mu), self.sigma)

        self.__plot_gaussian(initialize=True)
        self.__plot_function_distribution(initialize=True)

    def __plot_gaussian(self, initialize=False) -> None:
        """
        Plot the weight distribution to self.ax_weight.
        """
        xlim = self.ax_weight.set_xlim()
        ylim = self.ax_weight.set_ylim()
        extent = (xlim[0]+self.mu[0], xlim[1]+self.mu[0], ylim[0]+self.mu[1], ylim[1]+self.mu[1])
    
        if initialize:
            self.weight_plot = self.ax_weight.imshow(self.weight_density, cmap='Blues', aspect='auto', extent=extent)
            self.ax_weight.set_title('Weight space')
        else:
            self.weight_plot.set_extent(extent)

        self.weight_plot.figure.canvas.draw_idle()

    def __plot_function_distribution(self, initialize=False) -> None:
        """
        Calculate the function distribution from the weight distribution and plot it to self.ax_func.
        """
        n_samples = 100

        xlim = self.ax_func.set_xlim()
        ylim = self.ax_func.set_ylim()

        x = np.linspace(xlim[0], xlim[1], n_samples)
        y = np.linspace(ylim[0], ylim[1], n_samples)

        phi = np.stack((np.ones(x.size), x)).T

        # Project into function space.
        function_dist = self.project(phi)
        mu = function_dist.mu
        sigma = function_dist.sigma.diagonal()

        # TODO use x and phi to get actual x array

        densities = get_gaussian(-y[None, :].T, mu[None, :], sigma[None, :])

        if initialize:
            self.ax_func.set_title('Function space')
            self.func_plot = self.ax_func.imshow(densities, cmap='Blues', aspect='auto', extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
        else:
            self.func_plot.set_data(densities)
        
        self.func_plot.figure.canvas.draw_idle()


    def plot(self) -> None:
        """
        Plots the current weight and function space distributions to their assigned axes.
        """
        self.__plot_gaussian()
        self.__plot_function_distribution()

    def on_mouse_move(self, event) -> None:
        """
        Updates mu to be equal to the mouse position if mouse is currently dragging and 
        inside the weight space plot.
        """
        if self.dragging and event.xdata is not None and event.inaxes == self.ax_weight:
            self.mu[self.displayed_weights[0]] = event.xdata
            self.mu[self.displayed_weights[1]] = event.ydata
            self.plot()

    def on_mouse_button_down(self, event) -> None:
        """
        Start dragging if left button was pressed.
        """
        if event.button == MouseButton.LEFT:
            self.dragging = True

    def on_mouse_button_up(self, event) -> None:
        """
        Stops dragging if left button has been released.
        """
        if event.button == MouseButton.LEFT:
            self.dragging = False
            