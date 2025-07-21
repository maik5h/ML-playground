from features import FeatureVector, Feature
import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from typing import Union, List, SupportsIndex
from logging import warning
from config import Config


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
    
    def select_random_variables(self, indices: List[SupportsIndex]) -> List[np.array]:
        """
        Selects the random variables at the input indices and marginalizes out all other variables.
        Returns the corresponding mean vector mu and covariance matrix sigma.
        """
        mu = self.mu
        sigma = self.sigma

        # Find the indices to remove.
        rm_indices = [len(self.mu) - n - 1 for n in range(len(self.mu))]
        for i in indices:
            rm_indices.remove(indices[i])

        for idx in rm_indices:
            mu = np.delete(mu, idx)
            sigma = np.delete(sigma, idx, axis=0)
            sigma = np.delete(sigma, idx, axis=1)
        
        return mu, sigma

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

        # Phi is a FeatureVector, features is phi evaluated at Config.function_space_samples x-positions.
        self.phi: FeatureVector = None
        self._features: np.array = None

        # References to plots used to replace the data instead of replotting on every change.
        self._weight_plot = None
        self._func_plot = None

        self._setup(phi, weight_space_ax, function_space_ax)

    def _setup(self, phi: FeatureVector, weight_space_ax: plt.axis, function_space_ax: plt.axis) -> None:
        """
        Creates arrays of samples at which the weight and function space distributions are evaluated.
        Evaluates FeatureVector at these samples and initializes the plots.

        Attributes
        ----------

        phi: `FeatureVector`
            The feature vector associated with this Gaussian. 
        weight_space_ax: `matplotlib.pyplot.axis`
            Matplotlib axis to draw the weight space distribution onto.
        function_space_ax: `matplotlib.pyplot.axis`
            Matplotlib axis to draw the function space distribution onto.
        """
        self.phi = phi

        self._ax_weight: plt.axis = weight_space_ax
        self._ax_func: plt.axis = function_space_ax

        # Calculate samples at which weight and function space distributions are calculated.
        xlim = self._ax_weight.set_xlim()
        ylim = self._ax_weight.set_ylim()

        # As imshow is used instead of contourf() or similar, the y-axis has to be mirrored by convention.
        self.x1, self.x2 = np.meshgrid(
            np.linspace(xlim[0], xlim[1], Config.weight_space_samples),
            np.linspace(ylim[1], ylim[0], Config.weight_space_samples)
        )
        self._weight_samples = np.dstack((self.x1, self.x2))

        xlim = self._ax_func.set_xlim()
        ylim = self._ax_func.set_ylim()
        self._func_samples_x = np.linspace(xlim[0], xlim[1], Config.function_space_samples)
        self._func_samples_y = np.linspace(ylim[1], ylim[0], Config.function_space_samples)
   
        self._update_features()

        self._plot_weight_distribution(initialize=True)
        self._plot_function_distribution(initialize=True)

    def _update_features(self):
        """
        Evaluates the feature vector self.phi at self._func_samples_x The evaluated values are
        stored in self._features.
        Call everytime self.phi changes.
        """
        self._features = self.phi(self._func_samples_x)

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

        self.plot()
    
    def remove_feature(self, idx: int) -> None:
        # Remove dimension from parent Gaussian.
        self.remove_random_variable(idx)

        # Remove the feature from phi and update the array of features evaluated at x-values.
        self.phi.remove_feature(idx)
        self._update_features()

        self.plot()

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
        self.plot()

    def _plot_weight_distribution(self, initialize=False) -> None:
        """
        Plot the weight distribution to self._ax_weight.

        initialize == True:     Creates the plot object, must only be called once when initializing the
                                InteractiveGaussian instance.

        initialize == False:    Updates only the data of the existing plot.
        """
        active_mu, active_sigma = self.select_random_variables(self._active_idx)
        weight_density = multivariate_normal.pdf(self._weight_samples, active_mu, active_sigma)

        if initialize:
            xlim = self._ax_weight.set_xlim()
            ylim = self._ax_weight.set_ylim()
            extent = (xlim[0], xlim[1], ylim[0], ylim[1])

            self._weight_plot = self._ax_weight.imshow(weight_density, cmap='Blues', aspect='auto', extent=extent, vmax=Config.colormap_vmax)
            self._ax_weight.set_title('Weight space')
        else:
            self._weight_plot.set_data(weight_density)


        self._weight_plot.figure.canvas.draw_idle()

    def _plot_function_distribution(self, initialize=False) -> None:
        """
        Calculate the function distribution from the weight distribution and plot it to self._ax_func.
        The distribution over function values f(x) for a given x is Gaussian.
        """
        xlim = self._ax_func.set_xlim()
        ylim = self._ax_func.set_ylim()

        # Project into function space.
        function_dist = self.project(self._features)
        mu = function_dist.mu
        # TODO this is correct if sigma is diagonal, but what if it is not?
        sigma = function_dist.sigma.diagonal()

        densities = get_gaussian(self._func_samples_y[None, :].T, mu[None, :], sigma[None, :])

        if initialize:
            self._ax_func.set_title('Function space')
            self._func_plot = self._ax_func.imshow(densities, cmap='Blues', aspect='auto', extent=(xlim[0], xlim[1], ylim[0], ylim[1]), vmax=Config.colormap_vmax)
        else:
            self._func_plot.set_data(densities)
        
        self._func_plot.figure.canvas.draw_idle()

    def plot(self) -> None:
        """
        Plots the current weight and function space distributions to their assigned axes.
        """
        self._plot_weight_distribution()
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
            self.scale_sigma(1 - Config.mouse_wheel_sensitivity)
        elif event.button == 'down':
            self.scale_sigma(1 / (1 - Config.mouse_wheel_sensitivity))
