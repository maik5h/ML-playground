from features import FeatureVector
import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from typing import Union


# Densities can become very small very fast, so it is good to not capture the full range.
vmax = 0.3

scroll_sensitivity = 0.3

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
    def __init__(self, phi: FeatureVector, weight_space_ax: plt.axis, function_space_ax: plt.axis):
        """
        Initializes the Gaussian with mu = 0 and sigma = unit matching the dimension of the input feature vector.
        """
        Gaussian.__init__(self, np.zeros(len(phi)), np.eye(len(phi)))

        # Is the user currently dragging the weight distribution?
        self.dragging = False

        # The two weight distributions to display on the weight space plot.
        self.active_idx = [0, 1]

        # Phi is a FeatureVector, features is phi evaluated at n_samples x-positions.
        self.n_samples = 500
        self.phi: FeatureVector = None
        self.features: np.array = None

        # Matplotlib plots to 
        self.weight_plot = None
        self.func_plot = None

        self.setup(phi, weight_space_ax, function_space_ax)

    def setup(self, phi: FeatureVector, weight_space_ax: plt.axis, function_space_ax: plt.axis) -> None:
        """
        Initializes feature samples from FeatureVector.
        Sets the two axes to which the weight space and function space distributions are plotted.
        Creates a weight density array, which is reused as long as sigma does not change. Initializes
        the plots.

        phi(x):             A function that returns the feature vector of the model evaluated at position x,
                            where x can be a numpy array.
        weight_space_ax:    Matplotlib axis to draw the weight space distribution onto.
        function_space_ax:  Matplotlib axis to draw the function space distribution onto.
        """
        self.phi = phi

        # Evaluate the feature vector at n_samples x positions in the range of the function space plot.    
        xlim = function_space_ax.set_xlim()
        x = np.linspace(xlim[0], xlim[1], self.n_samples)
        self.features = phi(x)

        self.ax_weight = weight_space_ax
        self.ax_func = function_space_ax

        self.bake_weight_density()

        self.__plot_weight_distribution(initialize=True)
        self.__plot_function_distribution(initialize=True)

    def bake_weight_density(self):
        # Freeze an array of weight space densities for faster plotting.
        xlim = self.ax_weight.set_xlim()
        ylim = self.ax_weight.set_ylim()
        self.x1, self.x2 = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 50),
            np.linspace(ylim[0], ylim[1], 50)
        )
        pos = np.dstack((self.x1, self.x2))
        active_mu = (self.mu[self.active_idx[0]], self.mu[self.active_idx[1]])
        active_sigma = (self.sigma[self.active_idx[0], self.active_idx[0]],
                        self.sigma[self.active_idx[1], self.active_idx[1]])
        self.weight_density = multivariate_normal.pdf(pos, active_mu, active_sigma)

    def scale_sigma(self, scale: float) -> None:
        """
        Scales the sigma in all dimensions by scale and updates the weight space plot.
        """
        self.sigma *= scale
        self.bake_weight_density()
        self.plot(update_dist=True)


    def __plot_weight_distribution(self, initialize=False, update_dist=False) -> None:
        """
        Plot the weight distribution to self.ax_weight.
        Can be run in three differently GPU heavy modes (The value of update_dist is only
        important if initialize is False):

        initialize == True:     Creates the plot object, must only be called once when initializing the
                                InteractiveGaussian instance.

        update_dist == True:    Update the axis data of the existing plot to a new weight distribution
                                and set the axis extent to accomodate the current mu.

        update_dist == False:   Updates only the extent of the existing plot to accomodate changes in mu.
        """
        xlim = self.ax_weight.set_xlim()
        ylim = self.ax_weight.set_ylim()
        extent = (xlim[0]+self.mu[self.active_idx[0]],
                  xlim[1]+self.mu[self.active_idx[0]],
                  ylim[0]+self.mu[self.active_idx[1]],
                  ylim[1]+self.mu[self.active_idx[1]])
    
        if initialize:
            self.weight_plot = self.ax_weight.imshow(self.weight_density, cmap='Blues', aspect='auto', extent=extent)
            self.ax_weight.set_title('Weight space')
        elif update_dist:
            self.weight_plot.set_data(self.weight_density)
            self.weight_plot.set_extent(extent)
        else:
            self.weight_plot.set_extent(extent)

        self.weight_plot.figure.canvas.draw_idle()

    def __plot_function_distribution(self, initialize=False) -> None:
        """
        Calculate the function distribution from the weight distribution and plot it to self.ax_func.
        """
        xlim = self.ax_func.set_xlim()
        ylim = self.ax_func.set_ylim()

        x = np.linspace(xlim[0], xlim[1], self.n_samples)
        y = np.linspace(ylim[0], ylim[1], self.n_samples)

        # Project into function space.
        function_dist = self.project(self.features)
        mu = function_dist.mu
        sigma = function_dist.sigma.diagonal()

        # TODO use x and phi to get actual x array

        densities = get_gaussian(-y[None, :].T, mu[None, :], sigma[None, :])

        if initialize:
            self.ax_func.set_title('Function space')
            self.func_plot = self.ax_func.imshow(densities, cmap='Blues', aspect='auto', extent=(xlim[0], xlim[1], ylim[0], ylim[1]), vmax=vmax)
        else:
            self.func_plot.set_data(densities)
        
        self.func_plot.figure.canvas.draw_idle()


    def plot(self, update_dist=False) -> None:
        """
        Plots the current weight and function space distributions to their assigned axes.
        """
        self.__plot_weight_distribution(update_dist=update_dist)
        self.__plot_function_distribution()

    def on_mouse_move(self, event) -> None:
        """
        Updates mu to be equal to the mouse position if mouse is currently dragging and 
        inside the weight space plot.
        """
        if self.dragging and event.xdata is not None and event.inaxes == self.ax_weight:
            self.mu[self.active_idx[0]] = event.xdata
            self.mu[self.active_idx[1]] = event.ydata
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
    
    def on_scroll_event(self, event) -> None:
        """
        Adjusts all entries in sigma when scrolled over the weight space plot.
        """
        if event.button == 'up':
            self.scale_sigma(1 - scroll_sensitivity)
        elif event.button == 'down':
            self.scale_sigma(1 / (1 - scroll_sensitivity))
