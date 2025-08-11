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
    def condition(self, x: NDArray, y: NDArray, sigma: float) -> None:
        """
        Updates the model to represent the posterior distribution after
        observing values `y` at positions `x`. Sigma is the noise of
        the target distribution.
        Updates the plots associated with this model.
        """
        pass

    @abc.abstractmethod
    def get_likelihood(self, x: NDArray, y: NDArray) -> NDArray:
        """
        Returns the likelihood to observe a sample `y` at position `x`
        given the current model state.
        """
        pass
