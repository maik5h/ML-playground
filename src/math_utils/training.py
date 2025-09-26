from typing import Literal, Callable

import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from matplotlib.backend_bases import KeyEvent

from ..math_utils import TrainableModel
from ..config import Config
from ..gui_utils import create_button


# Tuple and type of all sampling orders that can be used for data
# loading.
SAMPLING_ORDERS = ('sequential', 'random', 'least likely')
SamplingOrder = Literal['sequential', 'random', 'least likely']


class DataLoader:
    """
    Class to store and load x and y data pairs.

    Attributes
    ----------

    x_data: `NDArray`
        Array of x data points.
    y_data: `NDArray`
        Array of y data points.
    order: `SamplingOrder`
        The order in which the datapoints are returned.
        - 'sequential': returns datapoints in the order they are
        stored in the arrays.
        - 'random': returns datapoints in random order.
        - 'least likely': returns the point which is the least likely
        given the current model prediction.
    """
    def __init__(self,
                 x_data: NDArray,
                 y_data: NDArray,
                 model: TrainableModel,
                 order: SamplingOrder,
                 batch_size: int = 1):
        self.x_data = x_data
        self.y_data = y_data
        self._model = model
        self._order = order
        self.batch_size = batch_size
        self._used_indices = []
        self._remaining_indices = np.arange(len(x_data))
    
    def set_order(self, order: SamplingOrder) -> None:
        """
        Sets the sampling order of this DataLoader. This resets the instance and starts the iteration over.
        """
        self._order = order
        self.reset()
    
    def set_data(self, x_data: NDArray, y_data: NDArray) -> None:
        """
        Sets the x and y data to the input data and resets internal state.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.reset()
    
    def get_used_values(self) -> tuple[NDArray, NDArray]:
        """
        Returns all x and y values that have already been returned by __next__.
        """
        return self.x_data[self._used_indices], self.y_data[self._used_indices]

    def get_remaining_values(self) -> tuple[NDArray, NDArray]:
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
            likelihood = self._model.get_likelihood(self.x_data[self._remaining_indices],
                                                    self.y_data[self._remaining_indices])

            # Created indexed array and sort it to find indices of lowest likelihood.
            likelihood = np.stack((likelihood, np.arange(len(likelihood)))).T
            likelihood = sorted(likelihood, key=lambda element: element[0])
            likelihood = np.array(likelihood)[:, 1].astype(np.int32)

            x = self.x_data[self._remaining_indices[likelihood[:self.batch_size]]]
            y = self.y_data[self._remaining_indices[likelihood[:self.batch_size]]]

            self._used_indices.extend(self._remaining_indices[likelihood[:self.batch_size]])
            if len(self._remaining_indices) >= self.batch_size:
                self._remaining_indices = np.delete(self._remaining_indices, likelihood[:self.batch_size])
            else:
                self._remaining_indices = []

            return x, y

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
    Adds a control panel to a given area on a matplotlib plot when
    initialized. The control panel offers access to the training
    procedure of a model, including the generation of target data,
    sampling and updating the model.
    """
    def __init__(self,
                 area: tuple[float, float, float, float],
                 model: TrainableModel,
                 data_generator: Callable[[], tuple[NDArray, NDArray]],
                 fig: Figure,
                 ax: Axes):
        """
        Parameters
        ----------

        area: `tuple[float, float, float, float]`
            Relative [left, bottom, width, height] coordinates on the
            figure where the InteractiveTrainer interface is displayed.
        model: `TrainableModel`
            The model that is being trained.
        data_generator: `Callable[[], tuple[NDArray, NDArray]]`
            A Callable that returns a tuple of target x and y samples.
        fig: `Figure`
            The figure to add the interface to.
        ax: `Axes`
            The axes to plot the target samples to.
        """
        self._model: TrainableModel = model
        self._data_generator = data_generator
        self._data_loader: DataLoader = DataLoader([], [], self._model, 'sequential')
        self._fig = fig
        self._ax = ax

        # References to the collections returned by the two scatter plots showing the datapoints that have been
        # used in training and the remaining ones. Must be removed before plotting again.
        self._collection_remaining = None
        self._collection_used = None

        # The animation that updates the plots regarding the model and
        # target data.
        self._anim = AnimationManager(self._create_animation)

        self._generate_target_data()

        # Create the interface buttons for:
        # - Generating new target samples replacing the previous data.
        # - Choosing the dataloader order.
        # - Starting the animated training.
        #
        # The buttons make up the whole accessible area, each having 0.3 of the total width.
        x, y, w, h = area
        self._sample_button = create_button(pos         = [x, y, 0.3*w, h],
                                            label       = 'Generate samples',
                                            on_clicked  = lambda event: self._generate_target_data())
        
        self._order_button = create_button(pos          = [x + 0.35*w, y, 0.3*w, h],
                                           label        = 'Sampling order:\nsequential',
                                           on_clicked   = lambda event: self._cycle_sampling_order())

        self._start_button = create_button(pos          = [x + 0.7*w, y, 0.3*w, h],
                                           label        = 'Start/ stop training\n(space  bar)',
                                           on_clicked   = lambda event: self._anim.play())
    
        self._fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._plot_target_samples()
        self._ax.legend()
    
    def _generate_target_data(self) -> None:
        """
        Generate random target data and replace the previous data.
        """
        self._anim.end()

        x_data, y_data = self._data_generator()
        self._data_loader.set_data(x_data, y_data)
        self._plot_target_samples()
    
    def _cycle_sampling_order(self) -> None:
        """
        Cycles through the available sampling orders of this instances DataLoader.
        """
        self._anim.end()

        # Update DataLoader sampling order.
        active_order = self._data_loader._order
        active_idx = SAMPLING_ORDERS.index(active_order)
        if active_idx == len(SAMPLING_ORDERS) - 1:
            active_idx = 0
        else:
            active_idx += 1
        
        # Update the order buttons text.
        self._data_loader.set_order(SAMPLING_ORDERS[active_idx])
        self._order_button.label.set_text(f'Sampling order:\n{SAMPLING_ORDERS[active_idx]}')
        self._plot_target_samples()

    def _plot_target_samples(self) -> None:
        """
        Plots the target samples to self._ax. Samples that have been used in training, i.e. samples that
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
        noise_sigma = Config.target_noise_amount * np.eye(len(x))
        # If only one datapoint is loaded, remove extra dimension created by np.eye().
        if len(x) == 1:
            noise_sigma = noise_sigma[0]

        # End the animation if the model requests so by returnin False.
        if not self._model.condition(x, y, noise_sigma):
            self._anim.end()

        self._plot_target_samples()

        # End the animation if last step has been reached.
        if len(self._data_loader.get_remaining_values()[0]) == 0:
            self._anim.end()

        return (self._collection_remaining, self._collection_used)

    def _create_animation(self) -> FuncAnimation:
        self._model.restart_training()
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

    def _on_key_press(self, event: KeyEvent) -> None:
        """
        Starts and stops the animated learning if the space bar has been pressed.
        """
        if event.key == ' ':
            self._anim.play()
