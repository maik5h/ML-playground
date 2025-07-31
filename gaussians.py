from features import FeatureVector, Feature
import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent, KeyEvent
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.stats import multivariate_normal
from typing import Union, Literal
from numpy.typing import NDArray
from logging import warning
from config import Config
import scipy as sc


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
    The probability density is Gaussian and characterized by the mean vector mu and the covariance matrix sigma.
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

    def condition(self, phi: NDArray, Y: NDArray, sigma: NDArray) -> None:
        """
        Updates this Gaussian to represent the conditional probability given a linear transformation phi,
        data Y and the noise amount on the data sigma.
        """
        # TODO the data noise sigma is required here. I dont like that as in reality the noise might be unknown.
        # Is there a way to avoid it?

        # If the data Y conatins only one datum, use simplified form of inference.
        if Y.shape == (1,):
            self.mu += ((self.sigma @ phi / (phi.T @ self.sigma @ phi + sigma ** 2)) * (Y - phi.T @ self.mu)).squeeze()
            self.sigma -= (self.sigma @ phi / (phi.T @ self.sigma @ phi + sigma ** 2)) @ (phi.T @ self.sigma)

        # If data has multiple points, use cholensky decomposition of the matrix
        # A = phi.T @ self.sigma @ phi + sigma ** 2 and solve linear equation instead of
        # explicitly calculating A^-1.
        else:
            fac = sc.linalg.cho_factor(phi.T @ self.sigma @ phi + sigma ** 2)

            self.mu += self.sigma @ phi @ sc.linalg.cho_solve(fac, (Y - phi.T @ self.mu))
            self.sigma -= self.sigma @ phi @ sc.linalg.cho_solve(fac, phi.T @ self.sigma)

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

        # Indicates if the user is currently dragging the weight distribution and if x, y or r keys are pressed.
        self._dragging = False
        self._key_pressed = {
            'x': False,
            'y': False,
            'r': False
        }

        # The two weight distributions to display on the weight space plot.
        self._active_idx = [0, 1]

        # Phi is a FeatureVector, features is phi evaluated at Config.function_space_samples x-positions.
        self.phi: FeatureVector = None
        self._features: NDArray = None

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
   
        self._setup_weight_buttons()

        self._update_features()

        self._plot_weight_distribution(initialize=True)
        self._plot_function_distribution(initialize=True)
    
    def _setup_weight_buttons(self) -> None:
        """
        Initializes the buttons serving as the weight space x and y labels.
        """
        # I do not care about proper alignment, it would be too tidious for a small project
        # like this and unnecessarily bloat the code. The position and size of the buttons
        # scales with the axis but apart from that it is rather arbitrary. 
        # Finished is better than perfect.

        # Instead of x and y labels, two Buttons are added to the weight space plot.
        pos = self._ax_weight.get_position()

        # The width of both buttons is determined from the width of the axis, for the x_button y-position
        # an arbitrary value of 10% of the lower axis y-position has been chosen.
        x_button_width = (pos.x1 - pos.x0) / 2
        x_button_height = 0.25 * x_button_width
        x_button_ax = plt.axes(((pos.x0 + pos.x1 - x_button_width) / 2,
                                pos.y0 * 0.1,
                                x_button_width,
                                x_button_height))

        self._weight_x_button = Button(x_button_ax, '')
        self._weight_x_button.on_clicked(lambda event: self._cycle_displayed_weight('x'))

        # Coordinates are relative, so width and height have to be scaled with the figures aspect ratio.
        aspect = 1 / 2
        y_button_x_offset = (pos.x1 - pos.x0) * 0.2
        y_button_ax = plt.axes((pos.x0 - y_button_x_offset,
                                (pos.y0 + pos.y1 - x_button_width / aspect) / 2,
                                x_button_height * aspect,
                                x_button_width / aspect))

        self._weight_y_button = Button(y_button_ax, '')
        self._weight_y_button.label.set_rotation(90)
        self._weight_y_button.on_clicked(lambda event: self._cycle_displayed_weight('y'))

    def _update_features(self) -> None:
        """
        Evaluates the feature vector self.phi at self._func_samples_x and stores the samples
        in self._features. Also updates the x and y label buttons of the weight space plot
        to display the correct description.
        Call everytime self.phi changes.
        """
        self._features = self.phi(self._func_samples_x)

        x_idx = self._active_idx[0]
        x_button_label = f'$w_{x_idx+1} ({self.phi[x_idx].get_expression()})$'
        self._weight_x_button.label.set_text(x_button_label)

        y_idx = self._active_idx[1]
        y_button_label = f'$w_{y_idx+1} ({self.phi[y_idx].get_expression()})$'
        self._weight_y_button.label.set_text(y_button_label)

    def add_feature(self, feature: Feature) -> None:
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
    
    def remove_feature(self, rm_idx: int) -> None:
        """
        Removes a feature from the FeatureVector associated with this instance and removes the
        corresponding weight from this distribution.

        Attributes
        ----------
        rm_idx: `int`
            The index of the feature and weight to be removed.
        """
        # If a feature is removed, all following features shift by one in the list. Make sure the
        # displayed features dont change when it happens.
        # To correctly update the currently displayed weights, find the weights that do not change.
        keep_index = (self._active_idx[0] < rm_idx, self._active_idx[1] < rm_idx)

        # If both indices change, offset both by one.
        if not keep_index[0] and not keep_index[1]:
            self._active_idx[0] -= 1
            self._active_idx[1] -= 1
        
        # If only one changes, move it but skip the other active index, if it occupies the next available
        # slot.
        elif not (keep_index[0] and keep_index[1]):
            if keep_index[0]:
                offset = 2 if self._active_idx[0] == self._active_idx[1] - 1 else 1
                self._active_idx[1] -= offset

            if keep_index[1]:
                offset = 2 if self._active_idx[1] == self._active_idx[0] - 1 else 1
                self._active_idx[0] -= offset
        
        # Take care of negative indices.
        for i in (0, 1):
            if self._active_idx[i] < 0:
                self._active_idx[i] += len(self.phi) - 1

        # Remove dimension from parent Gaussian.
        self.remove_random_variable(rm_idx)

        # Remove the feature from phi and update the array of features evaluated at x-values.
        self.phi.remove_feature(rm_idx)
        self._update_features()

        self.plot()

    def update_feature_parameter(self) -> None:
        """
        Updates the function space distribution after a feature parameter has changed.
        """
        self._update_features()
        self._plot_function_distribution()

    def _scale_sigma(self, factor: float, direction: tuple[bool, bool]) -> None:
        """
        Scales the sigma matrix entries concerning the currently displayed random variables.

        Attributes
        ----------

        factor: `float`
            The scaling factor applied to the affected entries.
        direction: `tuple[bool, bool]`
            Indicates whether to scale the entry corresponding to the currently displayed variables or not.
            direction[0] == True: scale along self._active_idx[0]
            direction[1] == True: scale along self._active_idx[1]
        """
        # Define a scaling matrix that scales the active weights only in the directions set to True.
        scale = np.eye(len(self.phi))
        i, j = self._active_idx[0], self._active_idx[1]
        scale[[i, j], [i, j]] = [factor if direction[0] else 1, factor if direction[1] else 1]

        self.sigma = scale @ self.sigma @ scale.T
        
        self.plot()
    
    def _rotate_sigma(self, angle: float) -> None:
        """
        Rotates the pdf with respect to the currently displayed random variables around their mean
        """
        # Create unity matrix and insert rotation matrix at concerned indices.
        rot = np.eye(len(self.phi))
        c, s = np.cos(angle), np.sin(angle)
        i, j = self._active_idx[0], self._active_idx[1]
        rot[[i, i, j, j], [i, j, i, j]] = [c, -s, s, c]

        self.sigma = rot @ self.sigma @ rot.T
        
        self.plot()
    
    def _reset_active_variables(self) -> None:
        """
        Resets the currently displayed random variables to mean zero and diagonal covariance one.
        """
        self.mu[self._active_idx[0]] = 0
        self.mu[self._active_idx[1]] = 0

        self.sigma[self._active_idx[0], self._active_idx[0]] = 1
        self.sigma[self._active_idx[1], self._active_idx[1]] = 1
        self.sigma[self._active_idx[0], self._active_idx[1]] = 0
        self.sigma[self._active_idx[1], self._active_idx[0]] = 0

        self.plot()

    def _plot_weight_distribution(self, initialize: bool = False) -> None:
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

    def _plot_function_distribution(self, initialize: bool = False) -> None:
        """
        Calculate the function distribution from the weight distribution and plot it to self._ax_func.
        """
        xlim = self._ax_func.set_xlim()
        ylim = self._ax_func.set_ylim()

        # To obtain the distribution over function values at a given x, the weights are multiplied with
        # the feature vector phi evaluated at that x: phi(x) @ w = phi_0(x) * w_0 + phi_1(x) * w_1 + ...

        # The product of _features with mu creates an array of size len(_features) with the
        # new mean for every x value corresponding to the entries in _features.
        mu = self._features @ self.mu

        # The variance of sigma transformed by a single feature is given by
        # new_sigma = feature @ sigma @ feature.T. However, since broadcasting is used
        # I deviate from this form and use an element wise multiplication followed
        # by a summation instead. This performs the multiplication and subsequent
        # summation otherwise done by the inner product for every of the stacked features.
        sigma = np.sum(self._features.T * (self.sigma @ self._features.T), axis=0)

        densities = get_gaussian(self._func_samples_y[None, :].T, mu, sigma)

        if initialize:
            self._ax_func.set_title('Function space')
            self._ax_func.set_xlabel('x')
            self._ax_func.set_ylabel('f(x)')
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

    def get_likelihood(self, x_data: NDArray, y_data: NDArray) -> NDArray:
        """
        Returns the likelihood to draw a sample y at position x in function space.
        """
        # Sample features at x_data.
        features = self.phi(x_data)

        # y data follows Gaussian distributions with own mu and sigma for every x value.
        mu = features @ self.mu
        sigma = np.sum(features.T * (self.sigma @ features.T), axis=0)

        likelihood = get_gaussian(y_data, mu, sigma)
        return likelihood
    
    def _cycle_displayed_weight(self, axis: Literal['x', 'y']) -> None:
        """
        Changes the weight displayed at idx (idx == 0: x-axis, idx == 1: y-axis) in the weight space plot.
        Cycles through available indices while skipping the one that is currently displayed on the other axis.
        """
        idx = 0 if axis == 'x' else 1
        # Increase active index by one and jump back to start if last index is surpassed.
        self._active_idx[idx] = (self._active_idx[idx] + 1) % len(self.mu)

        # If both axes show the same distribution, increase once again.
        if self._active_idx[0] == self._active_idx[1]:
            self._active_idx[idx] = (self._active_idx[idx] + 1) % len(self.mu)
        
        # Update the button label.
        n = self._active_idx[idx]
        label = f'$w_{n+1} ({self.phi[n].get_expression()})$'
        if idx == 0:
            self._weight_x_button.label.set_text(label)
        elif idx == 1:
            self._weight_y_button.label.set_text(label)
        self.plot()

    def on_mouse_move(self, event: MouseEvent) -> None:
        """
        Updates mu to be equal to the mouse position if mouse is currently dragging and 
        inside the weight space plot.
        """
        if self._dragging and event.xdata is not None and event.inaxes == self._ax_weight:
            self.mu[self._active_idx[0]] = event.xdata
            self.mu[self._active_idx[1]] = event.ydata
            self.plot()

    def on_mouse_button_down(self, event: MouseEvent) -> None:
        """
        Start dragging if left button was pressed. Resets the distribution of the currently displayed
        if doubleclicked.
        """
        if event.button == MouseButton.LEFT:
            self._dragging = True
        
        if event.dblclick:
            self._reset_active_variables()

    def on_mouse_button_up(self, event: MouseEvent) -> None:
        """
        Stops dragging if left button has been released.
        """
        if event.button == MouseButton.LEFT:
            self._dragging = False
        
    def on_key_pressed(self, event: KeyEvent) -> None:
        """
        Registers if keys of interest ('x', 'y' or 'r') have been pressed.
        """
        # If multiple keys are pressed at once, they are indicated as 'key1+key2+...'.
        keys = event.key.split('+')
        for key in keys:
            if key in self._key_pressed.keys():
                self._key_pressed[key] = True
    
    def on_key_released(self, event: KeyEvent) -> None:
        """
        Registers if keys of interest ('x', 'y', 'r') have been released.
        """
        # If multiple keys are released at once, they are indicated as 'key1+key2+...'.
        keys = event.key.split('+')
        for key in keys:
            if key in self._key_pressed.keys():
                self._key_pressed[key] = False
    
    def on_scroll_event(self, event: MouseEvent) -> None:
        """
        Either scales the distribution of the currently displayed variables
        - along x-direction if 'x' key is pressed,
        - along y-direction if 'y'-key is pressed,
        - along both directions if both 'x'- and 'y' key or none of them are pressed.

        or rotates the distribution if 'r'-key is pressed.
        """
        # If alt is pressed, only rotate regardless of other key states.
        if self._key_pressed['r']:
            angle = 0.1 * Config.mouse_wheel_sensitivity
            angle = angle if event.button == 'up' else -angle

            # Snap angle to some integer fraction of pi, so the initial state is always restored after
            # a finite amount of scroll events.
            angle = (np.pi) / (int(np.pi / angle))

            self._rotate_sigma(angle)

        else:
            if event.button == 'up':
                factor = 1 - 0.3 * Config.mouse_wheel_sensitivity
            elif event.button == 'down':
                factor = 1 / (1 - 0.3 * Config.mouse_wheel_sensitivity)

            # Scale both directions if no button is pressed and only the selected directions else.
            if not (self._key_pressed['x'] or self._key_pressed['y']):
                self._scale_sigma(factor, (True, True))
            else:
                self._scale_sigma(factor, (self._key_pressed['x'], self._key_pressed['y']))
