import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from matplotlib.backend_bases import KeyEvent
from typing import Literal, Callable
from config import Config
from features import *
from feature_controls import create_button
from gaussians import InteractiveGaussian
from time import sleep


# Tuple of all Feature classes.
available_features = (PolynomialFeature, SineFeature, CosineFeature)

# Tuple of all sampling orders.
data_loader_sampling_orders = ['sequential', 'random', 'least likely']

def generate_target_samples(n_samples: int, noise_amount: float) -> tuple[np.array, np.array]:
    """
    Generates a randomized selection of samples.

    Creates a function from the Features defined in features.py with weights and parameters within
    the range used by the gaussians.InteractiveGaussian plots and feature_controls.FeatureController
    interface. This makes sure that the target funciton can always be approached using the interface.
    Gaussian noise is added to the function values.

    Returns x- and y-values of the samples in separate arrays.

    Parameters
    ----------

    n_samples: `int`
        Number of samples to take from the target function.
    noise_amount: `float`
        Variance of the Gaussian noise added to the samples.
    """
    # Create a random target function using a FeatureVector.
    target_function = FeatureVector([])
    n_features = np.random.randint(1, 3)
    for _ in range(n_features):
        # Choose a parameter that lies within the available weight space.
        parameter = np.random.randint(0, Config.weight_space_xlim[1]*0.8)
        feature_type = np.random.randint(0, len(available_features))
        target_function.add_feature(available_features[feature_type](parameter))

    # Evaluate the target function at n_samples x positions. This returns an array with each
    # feature evaluated at x separately with shape=(n_samples, n_features).
    x_range = Config.function_space_xlim
    x_samples = np.linspace(x_range[0], x_range[1], n_samples)
    y_features = target_function(x_samples)

    # Create a weight for each feature and make sure all weights lie inside the plotted space.
    weights = np.random.rand(n_features) * Config.weight_space_xlim[1]
    y_samples = y_features @ weights

    y_samples += np.random.normal(0, scale=noise_amount, size=y_samples.shape)

    return x_samples, y_samples

class DataLoader:
    """
    Class to store and load x and y data pairs as returned by generate_target_samples.

    Attributes
    ----------

    x_data: `np.array`
        Array of x data points.
    y_data: `np.array`
        Array of y data points.
    order: `Literal['sequential', 'random', 'least likely']`
        The order in which the datapoints are returned.
        - 'sequential': returns datapoints in the order they are stored in the arrays.
        - 'random': returns datapoints in random order.
        - 'least likely': TODO return the point which is the least likely given the current model prediction.
    """
    def __init__(self, x_data: np.array, y_data: np.array, order: Literal['sequential', 'random', 'least likely'], batch_size: int = 1):
        self.x_data = x_data
        self.y_data = y_data
        self._order = order
        self.batch_size = batch_size
        self._used_indices = []
        self._remaining_indices = np.arange(len(x_data))
    
    def set_order(self, order: Literal['sequential', 'random', 'least likely']) -> None:
        """
        Sets the sampling order of this DataLoader. This resets the instance and starts the iteration over.
        """
        self._order = order
        self.reset()
    
    def set_data(self, x_data: np.array, y_data: np.array) -> None:
        """
        Sets the x and y data to the input data and resets internal state.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.reset()
    
    def get_used_values(self) -> tuple[np.array, np.array]:
        """
        Returns all x and y values that have already been returned by __next__.
        """
        return self.x_data[self._used_indices], self.y_data[self._used_indices]

    def get_remaining_values(self) -> tuple[np.array, np.array]:
        """
        Returns the values that have not been returned by __next__.
        """
        return self.x_data[self._remaining_indices], self.y_data[self._remaining_indices]
    
    def reset(self) -> None:
        """
        Resets the iteration state of this DataLoader. Does not change the data or sampling order.
        """
        self._used_indices = []
        self._remaining_indices = np.arange(len(self.x_data))

    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> tuple[float, float]:
        if len(self._remaining_indices) == 0:
            raise StopIteration

        if self._order == 'sequential':
            x = self.x_data[self._remaining_indices[0: self.batch_size]]
            y = self.y_data[self._remaining_indices[0: self.batch_size]]
            self._used_indices.extend(self._remaining_indices[0: self.batch_size])
            if len(self._remaining_indices) >= self.batch_size:
                self._remaining_indices = np.delete(self._remaining_indices, slice(0, self.batch_size))
            else:
                self._remaining_indices = []

            return x, y
        
        elif self._order == 'random':
            idx = np.random.randint(0, len(self._remaining_indices), size=self.batch_size)

            x = self.x_data[self._remaining_indices[idx]]
            y = self.y_data[self._remaining_indices[idx]]
            self._used_indices.extend(self._remaining_indices[idx])
            if len(self._remaining_indices) >= len(idx):
                self._remaining_indices = np.delete(self._remaining_indices, idx)
            else:
                self._remaining_indices = []

            return x, y
        
        elif self._order == 'least likely':
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.x_data)

class AnimationManager:
    """
    Class to manage a matplotlib FuncAnimation. Offers easier control to start, stop or start over from
    frame zero using the methods play() and end().
    """
    def __init__(self, get_animation: Callable[[], FuncAnimation]):
        """
        Parameters
        ----------

        get_animation: `callable[[], FuncAnimation]`
            A function that creates a FuncAnimation object with the desired properties when called.
        """
        self._get_animation = get_animation

        # Reference to a FuncAnimation object.
        self._animation: FuncAnimation = None

        # Restarts the animation from frame zero on the next play() call.
        self._restart_on_next_play = True

        # Indicates wether the animation is playing or paused.
        self._is_playing = True
    
    def end(self) -> None:
        """
        Ends the animation. Will cause it to start from scratch on the next play() call.
        """
        if self._animation is not None:
            self._animation.pause()
        self._restart_on_next_play = True
        self._is_playing = True

    def play(self) -> None:
        """
        Starts the animation if it is not currently playing or pauses/ resumes it if it is.
        """
        if self._restart_on_next_play:
            self._animation = self._get_animation()
            self._restart_on_next_play = False
        else:
            if self._is_playing:
                self._animation.pause()
                self._is_playing = False
            else:
                self._animation.resume()
                self._is_playing = True

class InteractiveTrainer:
    """
    Extends the Trainer class by methods to plot the target data, change the dataloader mode and start
    the training during runtime.
    """
    def __init__(self, area: tuple[float, float, float, float], gaussian: InteractiveGaussian, fig: plt.figure, ax: plt.axis):
        """
        Parameters
        ----------

        area: `tuple[float, float, float, float]`
            Relative XYXY coordinates on the figure where the IntaeractiveTrainer interface is displayed.
        gaussian: `InteractiveGaussian`
            The model that is being trained.
        fig: `matplotlib.pyplot.figure`
            The figure to add the interface to.
        ax: `matplotlib.pyplot.axis`
            The axis to plot the target samples to.
        """
        self.gaussian: InteractiveGaussian = gaussian
        self._data_loader: DataLoader = DataLoader([], [], 'sequential')
        self._fig = fig
        self._ax = ax

        # References to the collections returned by the two scatter plots showing the datapoints that have been
        # used in training and the remaining ones. Must be removed before plotting again.
        self._collection_remaining = None
        self._collection_used = None

        # The animation that updates the gaussian weights and the plot of target samples.
        self._anim = AnimationManager(self._create_animation)

        self._generate_target_data()

        # Create the interface buttons for:
        # - Generating new target samples replacing the previous data.
        # - Choosing the dataloader order.
        # - Starting the animated training.
        #
        # The buttons make up the whole accessible area, each having 0.3 of the total width.
        x = area[0]
        y = area[1]
        w = area[2] - area[0]
        h = area[3] - area[1]
        self._sample_button = create_button(pos         = [x, y, 0.3*w, h],
                                            label       = 'Generate samples',
                                            on_clicked  = lambda event: self._generate_target_data())
        
        self._order_button = create_button(pos          = [x + 0.35*w, y, 0.3*w, h],
                                           label        = 'Sampling order:\nsequential',
                                           on_clicked   = lambda event: self._cycle_sampling_order())

        self._start_button = create_button(pos          = [x + 0.7*w, y, 0.3*w, h],
                                           label        = 'Start/ stop training\n(space  bar)',
                                           on_clicked   = lambda event: self._anim.play())
    
        self._plot_target_samples()
        self._ax.legend()
    
    def _generate_target_data(self) -> None:
        """
        Generate random target data and replace the previous data.
        """
        self._anim.end()

        x_data, y_data = generate_target_samples(Config.number_target_samples, Config.target_noise_amount)
        self._data_loader.set_data(x_data, y_data)
        self._plot_target_samples()
    
    def _cycle_sampling_order(self) -> None:
        """
        Cycles through the available sampling orders of this instances DataLoader.
        """
        self._anim.end()

        # Update DataLoader sampling order.
        active_order = self._data_loader._order
        active_idx = data_loader_sampling_orders.index(active_order)
        if active_idx == len(data_loader_sampling_orders) - 1:
            active_idx = 0
        else:
            active_idx += 1
        
        # Update the order buttons text.
        self._data_loader.set_order(data_loader_sampling_orders[active_idx])
        self._order_button.label.set_text(f'Sampling order:\n{data_loader_sampling_orders[active_idx]}')
        self._plot_target_samples()

    def _plot_target_samples(self) -> None:
        """
        Plots the target samples to self.ax. Samples that have been used in training, i.e. samples that
        have been returned by self._data_loader, are displayed as small dots. Samples that werent used
        in training yet are large 'o's.
        """
        if self._collection_remaining is not None:
            self._collection_remaining.remove()
            self._collection_used.remove()

        x_remaining, y_remaining = self._data_loader.get_remaining_values()
        x_used, y_used = self._data_loader.get_used_values()

        self._collection_remaining = self._ax.scatter(x_remaining, y_remaining, marker='o', color='k', s=4, alpha=0.3, label='remaining samples')
        self._collection_used = self._ax.scatter(x_used, y_used, marker='.', color='k', s=2, alpha=0.3, label='used samples')

        self._ax.figure.canvas.draw_idle()
    
    def _training_step(self, _frame) -> tuple[PathCollection, PathCollection]:
        """
        Performs a single training step by explicitly drawing data using self._data_loader.__next__().
        Updates the target samples plot.
        """
        x, y = self._data_loader.__next__()

        # Update the Gaussians weights.
        phi = self.gaussian.phi(x).T
        noise_sigma = Config.target_noise_amount * np.eye(len(x))
        noise_sigma = 1 * np.eye(len(x))

        # If only one datapoint is loaded, remove extra dimension created by np.eye().
        if len(x) == 1:
            noise_sigma = noise_sigma[0]

        self.gaussian.condition(phi, y, noise_sigma)
        self.gaussian.plot()

        self._plot_target_samples()

        # End the animation if last step has been reached.
        if len(self._data_loader.get_remaining_values()[0]) == 0:
            self._anim.end()

        return (self._collection_remaining, self._collection_used)

    def _create_animation(self) -> FuncAnimation:
        self._data_loader.reset()
        data = self._data_loader
        animation = FuncAnimation(self._fig, self._training_step,
                                    init_func=lambda : self._training_step(0),
                                    frames=np.arange(len(data)),
                                    interval=Config.time_per_learning_step,
                                    repeat=False)
        self._ax.figure.canvas.draw()

        return animation

    def reset_training(self) -> None:
        """
        Resets the training state of this InteractiveTrainer.
        """
        self._anim.end()
        self._data_loader.reset()
        self._plot_target_samples()

    def on_key_press(self, event: KeyEvent) -> None:
        """
        Starts and stops the animated learning if the space bar has been pressed.
        """
        if event.key == ' ':
            self._anim.play()
