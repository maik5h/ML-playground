"""Classes to visualize and edit a ParametricGaussian model.

WeightSpaceGUI and FunctionSpacePlot visualize the weight space and
function space distribution respectively. WeightSpaceGUI can further
change the properties of the model by processing user inputs using
matplotlib callbacks.
Both classes are fully aware of the ParametricGaussian class, while
the latter only accesses their `update` method.
"""


from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.widgets import Button
from matplotlib.backend_bases import MouseButton, MouseEvent, KeyEvent
import numpy as np
from scipy.stats import multivariate_normal

from .model import ParametricGaussian, StateInfo
from ..config import Config


class WeightSpaceGUI:
    """
    Class to handle plotting and user inputs regarding the weight space
    distribution of a ParametricGaussian model.

    Two of the Gaussian random variables are displayed on a given axis
    and can be edited through different user inputs.
    """
    def __init__(self, model: ParametricGaussian, fig: Figure, ax: Axes):
        self._model = model
        self._ax = ax

        # Connect the update method of this instance to the model to
        # register this instance as an 'observer'.
        self._model.notify_weight_gui = self.update

        # The indices of the two currently displayed random variables.
        self._active_idx = [0, 1]

        # Indicates if the user is currently dragging on self._ax and
        # if x, y or r keys are pressed.
        self._dragging = False
        self._key_pressed = {
            'x': False,
            'y': False,
            'r': False
        }

        # Calculate samples at which weight space distributions are
        # calculated.
        xlim = self._ax.set_xlim()
        ylim = self._ax.set_ylim()
        # As imshow is used instead of contourf() or similar, the
        # y-axis has to be mirrored by convention.
        self.x1, self.x2 = np.meshgrid(
            np.linspace(xlim[0], xlim[1], Config.weight_space_samples),
            np.linspace(ylim[1], ylim[0], Config.weight_space_samples)
        )
        self._weight_samples = np.dstack((self.x1, self.x2))
        self._plot_lims = (xlim[0], xlim[1], ylim[0], ylim[1])

        self._setup_weight_buttons()

        # Connect matplotlib callbacks.
        fig.canvas.mpl_connect('button_press_event', self._on_mouse_button_down)
        fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        fig.canvas.mpl_connect('button_release_event', self._on_mouse_button_up)
        fig.canvas.mpl_connect('key_press_event', self._on_key_pressed)
        fig.canvas.mpl_connect('key_release_event', self._on_key_released)
        fig.canvas.mpl_connect('scroll_event', self._on_scroll_event)

        self._plot(initialize=True)

    def _setup_weight_buttons(self) -> None:
        """
        Initializes the buttons serving as the weight space x and y
        labels.
        """
        # I do not care about proper alignment, it would be too tidious
        # for a small project like this and unnecessarily bloat the
        # code. The position and size of the buttons scales with the
        # axis but apart from that it is rather arbitrary. 
        # Finished is better than perfect.

        # Instead of x and y labels, two Buttons are added to the
        # weight space plot.
        pos = self._ax.get_position()

        # The width of both buttons is determined from the width of the
        # axis, for the x_button y-position an arbitrary value of 10%
        # of the lower axis y-position has been chosen.
        x_button_width = (pos.x1 - pos.x0) / 2
        x_button_height = 0.25 * x_button_width
        x_button_ax = plt.axes(((pos.x0 + pos.x1 - x_button_width) / 2,
                                pos.y0 * 0.1,
                                x_button_width,
                                x_button_height))
        self._weight_x_button = Button(x_button_ax, '')
        self._weight_x_button.on_clicked(lambda event: self._cycle_displayed_weight('x'))

        # Coordinates are relative, so width and height have to be
        # scaled with the figures aspect ratio.
        aspect = 1 / 2
        y_button_x_offset = (pos.x1 - pos.x0) * 0.2
        y_button_ax = plt.axes((pos.x0 - y_button_x_offset,
                                (pos.y0 + pos.y1 - x_button_width / aspect) / 2,
                                x_button_height * aspect,
                                x_button_width / aspect))
        self._weight_y_button = Button(y_button_ax, '')
        self._weight_y_button.label.set_rotation(90)
        self._weight_y_button.on_clicked(lambda event: self._cycle_displayed_weight('y'))

        self._update_label_text()

    def _update_label_text(self) -> None:
        """
        Sets the labels of the x- and y-label buttons to match the
        coresponding features.
        """
        n = self._active_idx[0]
        label = f'$w_{n+1} ({self._model.phi[n].get_expression()})$'
        self._weight_x_button.label.set_text(label)

        n = self._active_idx[1]
        label = f'$w_{n+1} ({self._model.phi[n].get_expression()})$'
        self._weight_y_button.label.set_text(label)

    def _plot(self, initialize: bool = False) -> None:
        """
        Plot the weight distribution to self._ax.

        initialize == True: Creates the AxesImage object, must only be
                            called once when initializing the
                            ParametricGaussian instance.

        initialize == False: Updates only the data of the existing
        AxesImage.
        """
        mu, sigma = self._model.select_random_variables(self._active_idx)
        weight_density = multivariate_normal.pdf(self._weight_samples, mu, sigma)

        if initialize:
            plot_kwargs = {'cmap': 'Blues',
                           'aspect': 'auto',
                           'extent': self._plot_lims,
                           'vmax': Config.colormap_vmax}
            
            self._ax_img = self._ax.imshow(weight_density, **plot_kwargs)
            self._ax.set_title('Weight space')
        else:
            self._ax_img.set_data(weight_density)

        self._ax_img.figure.canvas.draw_idle()

    def _process_removed_feature(self, rm_idx: int) -> None:
        """
        Updates the active indices such that the same variables stay
        active if possible.
        """
        # If a feature is removed, all following features shift by one
        # in the list. Make sure the displayed features dont change
        # when it happens. To correctly update the currently displayed
        # weights, find the weights that do not change.
        keep_index = (self._active_idx[0] < rm_idx, self._active_idx[1] < rm_idx)

        # If both indices change, offset both by one.
        if not keep_index[0] and not keep_index[1]:
            self._active_idx[0] -= 1
            self._active_idx[1] -= 1

        # If only one changes, move it but skip the other active index,
        # if it occupies the next available slot.
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
                self._active_idx[i] += len(self._model.phi)

    def _cycle_displayed_weight(self, axis: Literal['x', 'y']) -> None:
        """
        Changes the weight displayed at the `x` or `y` axis in the
        weight space plot. Cycles through available indices while
        skipping the one that is currently displayed on the other axis.

        This does not change the model state, only its representation
        on the plot.
        """
        idx = 0 if axis == 'x' else 1
        # Increase active index by one and jump back to start if last
        # index is surpassed.
        self._active_idx[idx] = (self._active_idx[idx] + 1) % len(self._model.mu)

        # If both axes show the same distribution, increase once again.
        if self._active_idx[0] == self._active_idx[1]:
            self._active_idx[idx] = (self._active_idx[idx] + 1) % len(self._model.mu)

        # Update the button label.
        self._update_label_text()
        self._plot()

    def update(self, info: StateInfo) -> None:
        """
        Updates the weight space plot after self._model has changed.
        This method must be made accessible inside self._model in order
        to notify this instance of changes.
        """
        if info.rm_feature is not None:
            self._process_removed_feature(info.rm_feature)
        if info.update_labels:
            self._update_label_text()
        if info.update_plot:
            self._plot()

    def _on_mouse_move(self, event: MouseEvent) -> None:
        """
        Updates mu to be equal to the mouse position if mouse is
        currently dragging and  inside the weight space plot.
        """
        if self._dragging and event.xdata is not None and event.inaxes == self._ax:
            self._model.set_mean(self._active_idx, [event.xdata, event.ydata])

    def _on_mouse_button_down(self, event: MouseEvent) -> None:
        """
        Start dragging if left button was pressed. Resets the
        distribution of the currently displayed if doubleclicked.
        """
        if event.button == MouseButton.LEFT:
            self._dragging = True

        if event.dblclick:
            self._model.reset_active_variables(self._active_idx)

    def _on_mouse_button_up(self, event: MouseEvent) -> None:
        """
        Stops dragging if left button has been released.
        """
        if event.button == MouseButton.LEFT:
            self._dragging = False

    def _on_key_pressed(self, event: KeyEvent) -> None:
        """
        Registers if keys of interest ('x', 'y' or 'r') have been
        pressed.
        """
        # If multiple keys are pressed at once, they are indicated as
        # 'key1+key2+...'.
        keys = event.key.split('+')
        for key in keys:
            if key in self._key_pressed.keys():
                self._key_pressed[key] = True

    def _on_key_released(self, event: KeyEvent) -> None:
        """
        Registers if keys of interest ('x', 'y', 'r') have been
        released.
        """
        # If multiple keys are released at once, they are indicated as
        # 'key1+key2+...'.
        keys = event.key.split('+')
        for key in keys:
            if key in self._key_pressed.keys():
                self._key_pressed[key] = False

    def _on_scroll_event(self, event: MouseEvent) -> None:
        """
        Can scale or rotate the distribution of the currently displayed
        variables.

        Checks for the following key press events:

        `x`: scale distribution only along x-direction.
        `y`: scale distribution only along y-direction.
        `r`: rotate distribution around its center.

        Scales along both x- and y-direction if both or none of the `x`
        and `y` keys are pressed.
        """
        # If `r` is pressed, rotate regardless of other key states.
        if self._key_pressed['r']:
            angle = 0.1 * Config.mouse_wheel_sensitivity
            angle = angle if event.button == 'up' else -angle

            # Snap angle to some integer fraction of pi, so the initial
            # state is always restored after a finite amount of scroll
            # events.
            angle = np.pi / int(np.pi / angle)

            self._model.rotate_sigma(angle, self._active_idx)

        else:
            if event.button == 'up':
                factor = 1 - 0.3 * Config.mouse_wheel_sensitivity
            elif event.button == 'down':
                factor = 1 / (1 - 0.3 * Config.mouse_wheel_sensitivity)

            # Scale only the selected indices. Further do only scale in
            # x- or y-direction if one of the x or y keys is pressed.
            # If both or none are pressed, scale both active indices.
            if self._key_pressed['x'] and not self._key_pressed['y']:
                scale_indices = [self._active_idx[0],]
            elif self._key_pressed['y'] and not self._key_pressed['x']:
                scale_indices = [self._active_idx[1],]
            else:
                scale_indices = self._active_idx

            self._model.scale_sigma(factor, scale_indices)


class FunctionSpacePlot:
    def __init__(self, model: ParametricGaussian, ax: Axes):
        self._model = model
        self._model.notify_func_gui = self.update
        self._ax = ax
        self._active_idx = [0, 1]

        # Prepare a grid to evaluate the function space density at.
        xlim = ax.set_xlim()
        ylim = ax.set_ylim()
        self._x = np.linspace(xlim[0], xlim[1], Config.function_space_samples)
        self._y = np.linspace(ylim[1], ylim[0], Config.function_space_samples)
        self._plot_lims = (xlim[0], xlim[1], ylim[0], ylim[1])

        self._plot(initialize=True)

    def update(self, info: StateInfo) -> None:
        """
        Updates the function space plot after self._model has changed.
        This method must be made accessible inside self._model in order
        to notify this instance of changes.
        """
        if info.update_plot:
            self._plot()

    def _plot(self, initialize=False) -> None:
        """
        Plot the function distribution to self._ax.

        initialize == True: Creates the AxesImage object, must only be
                            called once when initializing the
                            ParametricGaussian instance.

        initialize == False: Updates only the data of the existing
        AxesImage.
        """
        density = self._model.get_function_space_density(self._x, self._y)
        if initialize:
            self._ax.set_title('Function space')
            self._ax.set_xlabel('x')
            self._ax.set_ylabel('f(x)')

            plot_kwargs = {'cmap': 'Blues',
                           'aspect': 'auto',
                           'extent': self._plot_lims,
                           'vmax': Config.colormap_vmax}

            self._ax_img = self._ax.imshow(density, **plot_kwargs)
        else:
            self._ax_img.set_data(density)

        self._ax_img.figure.canvas.draw_idle()
