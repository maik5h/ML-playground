import abc

from numpy.typing import NDArray


class TrainableModel(metaclass=abc.ABCMeta):
    """
    Base class for models that can be conditioned on data. They must
    implement a `condition` method to update the model and a
    `get_likelihood` method in order to allow likelihood based
    sampling.
    """
    @abc.abstractmethod
    def condition(self, x: NDArray, y: NDArray) -> bool:
        """
        Updates the model to represent the posterior distribution after
        observing values `y` at positions `x`.
        Returns False to stop the training after this step, True else.
        """
        pass

    @abc.abstractmethod
    def get_likelihood(self, x: NDArray, y: NDArray) -> NDArray:
        """
        Returns the likelihood to observe a sample `y` at position `x`
        given the current model state.
        """
        pass

    @abc.abstractmethod
    def restart_training(self) -> None:
        """Notifies the model that a new training session has been
        started.
        The consequences are subject to the individual implementations.
        """
