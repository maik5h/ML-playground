import numpy as np
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent, MouseButton

from .model import ConditionalGaussianProcess
from ..math_utils import get_gaussian


# Arguments for covariance lines in GP function space plots.
PLOT_KWARGS = {
    'linestyle': '--',
    'color': 'tab:blue',
    'alpha': 0.8,
    'linewidth': 0.5
}


class GPFunctionSpacePlot:
    """Plots the functions space distribution of a Gaussian process.

    Displays the probability density of function values in shades of
    blue. The mean and the 2-std limits are displayed as solid/ dashed
    blue lines.
    """
    def __init__(
            self,
            ax: Axes,
            model: ConditionalGaussianProcess,
            resolution: int,
            vmax: float
        ):
        self._ax = ax
        self._model = model
        self._vmax = vmax

        xlim = self._ax.set_xlim()
        ylim = self._ax.set_ylim()
        self._x = np.linspace(xlim[0], xlim[1], resolution)
        self._y = np.linspace(ylim[0], ylim[1], resolution)

        self._ax.set_title('Function distribution')

        self._model.add_observer_call(self.plot)

        self.plot(initialize=True)

    def plot(self, initialize: bool = False) -> None:
        mu = self._model.posterior.get_mean(self._x)
        sigma = self._model.posterior.get_sigma(self._x)
        # Small values might end up negative due to machine precision.
        # Enforce a small positive minimum.
        sigma = np.where(sigma > 1e-5, sigma, 1e-5)

        densities = get_gaussian(self._y[:, None], mu, sigma)
        densities = np.flip(densities, axis=0)

        if initialize:
            xlim = self._ax.set_xlim()
            ylim = self._ax.set_ylim()

            self._ax.set_xlabel('x')
            self._ax.set_ylabel('f(x)')
            self._plot = self._ax.imshow(
                densities,
                cmap='Blues',
                aspect='auto',
                extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                vmax=self._vmax
            )
            self._mean_lines = self._ax.plot(self._x, mu, color='tab:blue', linewidth=0.5)
            self._cov_lines_1 = self._ax.plot(self._x, mu + 2*np.sqrt(sigma), **PLOT_KWARGS)
            self._cov_lines_2 = self._ax.plot(self._x, mu - 2*np.sqrt(sigma), **PLOT_KWARGS)

        else:
            for line in self._mean_lines:
                line.remove()
            for line in self._cov_lines_1:
                line.remove()
            for line in self._cov_lines_2:
                line.remove()
            self._plot.set_data(densities)
            self._mean_lines = self._ax.plot(self._x, mu, color='tab:blue', linewidth=0.5)
            self._cov_lines_1 = self._ax.plot(self._x, mu + 2*np.sqrt(sigma), **PLOT_KWARGS)
            self._cov_lines_2 = self._ax.plot(self._x, mu - 2*np.sqrt(sigma), **PLOT_KWARGS)

        self._plot.figure.canvas.draw_idle()


class GPKernelPlot:
    """Plots the kernel function of a Gaussian process.
    """
    def __init__(
            self,
            ax: Axes,
            model: ConditionalGaussianProcess,
            resolution: int,
            vmax: float
        ):
        self._ax = ax
        self._model = model
        self._vmax = vmax
        self._model.add_observer_call(self.plot)
        self._xlim = self._ax.set_xlim()

        # Plot kernel function in a range slightly larger than the
        # range in which samples are generated.
        self._x = np.linspace(self._xlim[0], self._xlim[1], resolution)
        self.plot(initialize=True)

    def plot(self, initialize: bool = False) -> None:
        kernel_func = self._model.posterior._kernel(self._x, self._x)
        kernel_func = np.flip(kernel_func, axis=0)

        if initialize:
            self._ax_img = self._ax.imshow(
                kernel_func,
                extent=[self._xlim[0], self._xlim[1], self._xlim[0], self._xlim[1]],
                cmap='Blues',
                vmax=self._vmax
            )
            self._ax.set_title(r'posterior kernel $k_{ab}$')
            self._ax.set_xlabel('a')
            self._ax.set_ylabel('b')
        else:
            self._ax_img.set_data(kernel_func)

        self._ax.figure.canvas.draw_idle()


class GPFunctionSamples:
    """Displays random samples from a Gaussian process on a plot.
    """
    def __init__(
            self,
            ax: Axes,
            model: ConditionalGaussianProcess,
            number_samples: int,
        ):
        """Parameters
        ----------

        ax: `matplotlib Axes`
            Axes to draw the samples on.
        model: `InteractiveGaussianProcess`
            The model from which samples are drawn.
        number_samples: `int`
            The number of samples to be displayed at once.
        """
        self._ax = ax
        self._model = model
        self._number_samples = number_samples

        # Always update the plot when the model state changes.
        self._model.add_observer_call(self.plot)

        self._gen = np.random.Generator(np.random.PCG64(seed=5))

        xlim = self._ax.set_xlim()
        self._x = np.linspace(xlim[0], xlim[1], 300)

        # Plot lines are stored here, so they can be removed
        # and replaced for each plot() call.
        self._lines: list[list[Line2D]] = []

        self._ax.figure.canvas.mpl_connect('button_press_event', self._on_mouse_button_down)

        self.plot()

    def plot(self) -> None:
        """Draws random samples from the given model and updates the plot."""
        # Draw samples from GP.
        mu = self._model.posterior.get_mean(self._x)
        sigma = self._model.posterior._kernel(self._x, self._x)

        samples = self._gen.multivariate_normal(
            mean=mu,
            cov=sigma,
            size=self._number_samples
        )

        # Remove previous lines from plot.
        for list in self._lines:
            for line in list:
                line.remove()
        self._lines.clear()

        # If multiple samples are plotted, decrease their alpha value
        # stepwise down to 0.2 to make them easier distinguishable.
        alpha = 0.8
        for sample in samples:
            lines = self._ax.plot(self._x, sample, color='tab:purple', linewidth=0.5, alpha=alpha)
            self._lines.append(lines)
            alpha -= 0.6 / self._number_samples

        self._ax.figure.canvas.draw_idle()

    def _on_mouse_button_down(self, event: MouseEvent) -> None:
        """Draw a new set of example samples from the model if left
        mouse button was clicked while the cursor was on the plot.
        """
        if event.button == MouseButton.LEFT and event.inaxes == self._ax:
            self.plot()
