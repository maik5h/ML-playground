from features import FeatureVector, Feature
import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from typing import Union
from logging import warning


# Densities can become very small very fast, so it is good to not capture the full range.
vmax = 0.3

scroll_sensitivity = 0.3

def get_gaussian(x, mu, sigma) -> Union[float, np.array]:
    """
    Returns the value of a Gaussian function with mean mu and standard deviation sigma at value x.
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


    # def condition(self, A: np.array, y: np.array):
    #     dec = linalg.cho_factor(self.sigma @ A.T @ (A @ self.sigma @ A.T))

    #     mu = linalg.cho_solve(dec, (y - A @ self.mu))
    #     sigma = linalg.cho_solve(dec, A @ self.sigma)

    #     return Gaussian(mu, sigma)


class InteractiveGaussian(Gaussian):
    """
    Extends the Gaussian class by methods to plot the distribution and update parameters based on matplotlib callbacks.
    A feature vector phi must be forwarded to objects of this class. The random variables forming this distribution are
    interpreted as weights being multiplied with this feature vector.
    
    Takes two axes, on which it will plot:
    - The distribution over two of its random variables.
    - The distribution over function values corresponding to phi.
    """
    def __init__(self, phi: FeatureVector, weight_space_ax: plt.axis, function_space_ax: plt.axis):
        """
        Initializes the Gaussian with mu = 0 and sigma = unit matching the dimension of the input feature vector.
        """
        if len(phi) < 2:
            raise RuntimeError('Feature vector must have at least two elements.')

        super().__init__(np.zeros(len(phi)), np.eye(len(phi)))

        # Indicates if the user is currently dragging the weight distribution.
        self._dragging = False

        # The two weight distributions to display on the weight space plot.
        self._active_idx = [0, 1]

        # Phi is a FeatureVector, features is phi evaluated at n_samples x-positions.
        self._n_samples = 500
        self.phi: FeatureVector = None
        self._features: np.array = None

        # References to plots used to replace the data instead of replotting on every change.
        self._weight_plot = None
        self._func_plot = None

        self._setup(phi, weight_space_ax, function_space_ax)

    def _setup(self, phi: FeatureVector, weight_space_ax: plt.axis, function_space_ax: plt.axis) -> None:
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

        self._ax_weight: plt.axis = weight_space_ax
        self._ax_func: plt.axis = function_space_ax
   
        self._update_features()

        self.bake_weight_density()

        self._plot_weight_distribution(initialize=True)
        self._plot_function_distribution(initialize=True)

    # TODO i dont think "baking" is necessary at all, it should plot sufficiently fast in real time.
    # It also seems bug prone and looks odd, as the pixel of th graph do not match the grid of the plot.
    def bake_weight_density(self):
        # Freeze an array of weight space densities for faster plotting.
        xlim = self._ax_weight.set_xlim()
        ylim = self._ax_weight.set_ylim()
        self.x1, self.x2 = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 50),
            np.linspace(ylim[0], ylim[1], 50)
        )
        pos = np.dstack((self.x1, self.x2))
        active_mu = (self.mu[self._active_idx[0]], self.mu[self._active_idx[1]])
        active_sigma = (self.sigma[self._active_idx[0], self._active_idx[0]],
                        self.sigma[self._active_idx[1], self._active_idx[1]])
        self.weight_density = multivariate_normal.pdf(pos, active_mu, active_sigma)

    def _update_features(self):
        """
        Evaluates the feature vector self.phi at different x-positions. self._n_samples positions
        are chosen from within the plotted function space interval. The evaluated values are
        stored in self._features.
        Call everytime self.phi changes.
        """
        xlim = self._ax_func.set_xlim()
        x = np.linspace(xlim[0], xlim[1], self._n_samples)
        self._features = self.phi(x)

    def add_feature(self, feature: Feature):
        """
        Adds a random variable to this Gaussian which corresponds to the input Feature. The plots
        are updated accordingly.
        """
        # Add dimension to parent Gaussian.
        self.add_random_variable()

        # Add the feature to phi and update the array of features evaluated at x-values.
        self.phi.add_feature(feature)
        self._update_features()

        self.bake_weight_density
        self.plot(update_dist=True)
    
    def remove_feature(self, idx: int) -> None:
        # Remove dimension from parent Gaussian.
        self.remove_random_variable(idx)

        # Remove the feature from phi and update the array of features evaluated at x-values.
        self.phi.remove_feature(idx)
        self._update_features()

        self.bake_weight_density
        self.plot(update_dist=True)

    def update_feature_parameter(self) -> None:
        """
        Updates the function space distribution after a feature parameter has changed.
        """
        self._update_features()
        self._plot_function_distribution()


    def scale_sigma(self, scale: float) -> None:
        """
        Scales the sigma in all dimensions by scale and updates the weight space plot.
        """
        self.sigma *= scale
        self.bake_weight_density()
        self.plot(update_dist=True)


    def _plot_weight_distribution(self, initialize=False, update_dist=False) -> None:
        """
        Plot the weight distribution to self._ax_weight.
        Can be run in three differently GPU heavy modes (The value of update_dist is only
        important if initialize is False):

        initialize == True:     Creates the plot object, must only be called once when initializing the
                                InteractiveGaussian instance.

        update_dist == True:    Updates the axis data of the existing plot to a new weight distribution
                                and set the axis extent to accomodate the current mu.

        update_dist == False:   Updates only the extent of the existing plot to accomodate changes in mu.
        """
        xlim = self._ax_weight.set_xlim()
        ylim = self._ax_weight.set_ylim()

        extent = (xlim[0]+self.mu[self._active_idx[0]],
                  xlim[1]+self.mu[self._active_idx[0]],
                  ylim[0]+self.mu[self._active_idx[1]],
                  ylim[1]+self.mu[self._active_idx[1]])
    
        if initialize:
            self._weight_plot = self._ax_weight.imshow(self.weight_density, cmap='Blues', aspect='auto', extent=extent)
            self._ax_weight.set_title('Weight space')
        elif update_dist:
            self._weight_plot.set_data(self.weight_density)
            self._weight_plot.set_extent(extent)
        else:
            self._weight_plot.set_extent(extent)

        self._weight_plot.figure.canvas.draw_idle()

    def _plot_function_distribution(self, initialize=False) -> None:
        """
        Calculate the function distribution from the weight distribution and plot it to self._ax_func.
        The distribution over function values f(x) for a given x is Gaussian.
        """
        xlim = self._ax_func.set_xlim()
        ylim = self._ax_func.set_ylim()
        y = np.linspace(ylim[0], ylim[1], self._n_samples)

        # Project into function space.
        function_dist = self.project(self._features)
        mu = function_dist.mu
        # TODO this is correct if sigma is diagonal, but what if it is not?
        sigma = function_dist.sigma.diagonal()

        densities = get_gaussian(-y[None, :].T, mu[None, :], sigma[None, :])

        if initialize:
            self._ax_func.set_title('Function space')
            self._func_plot = self._ax_func.imshow(densities, cmap='Blues', aspect='auto', extent=(xlim[0], xlim[1], ylim[0], ylim[1]), vmax=vmax)
        else:
            self._func_plot.set_data(densities)
        
        self._func_plot.figure.canvas.draw_idle()

    def plot(self, update_dist=False) -> None:
        """
        Plots the current weight and function space distributions to their assigned axes.
        """
        self._plot_weight_distribution(update_dist=update_dist)
        self._plot_function_distribution()

    def on_mouse_move(self, event) -> None:
        """
        Updates mu to be equal to the mouse position if mouse is currently dragging and 
        inside the weight space plot.
        """
        if self._dragging and event.xdata is not None and event.inaxes == self._ax_weight:
            self.mu[self._active_idx[0]] = event.xdata
            self.mu[self._active_idx[1]] = event.ydata
            self.plot()

    def on_mouse_button_down(self, event) -> None:
        """
        Start dragging if left button was pressed.
        """
        if event.button == MouseButton.LEFT:
            self._dragging = True

    def on_mouse_button_up(self, event) -> None:
        """
        Stops dragging if left button has been released.
        """
        if event.button == MouseButton.LEFT:
            self._dragging = False
    
    def on_scroll_event(self, event) -> None:
        """
        Adjusts all entries in sigma when scrolled over the weight space plot.
        """
        if event.button == 'up':
            self.scale_sigma(1 - scroll_sensitivity)
        elif event.button == 'down':
            self.scale_sigma(1 / (1 - scroll_sensitivity))
