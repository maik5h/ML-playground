from typing import Optional
from pathlib import Path
import yaml
from logging import warning


class GPRegressionConfig:
    # ----- Visuals -----
    # Number of pixels used to display the kernel function.
    kernel_resolution: int = 100

    # Number of pixels used to display the function space distribution.
    function_space_resolution: int = 300

    # Number of example functions drawn from the Gaussian process.
    number_function_samples: int = 1

    # Limits of function space plots.
    function_space_xlim: list[float] = [-5, 5]
    function_space_ylim: list[float] = [-10, 10]
    
    # The vmax value of both plots.
    colormap_vmax: float = 1


    # ----- learning -----
    # The total number of samples generated.
    number_target_samples: int = 10

    # Variance of the Gaussian noise added to the target samples.
    target_noise_amount: float = 1

    # Variance of the Gaussian noise assumed by the model.
    model_noise_amount: float = 1

    # The time in milliseconds between weight updates. This is the
    # minimum time a step will take, the actual time is limited by the
    # time it takes to update the weights and to plot the
    # distributions.
    time_per_learning_step: float = 300

    # Batch size of the dataloader.
    samples_per_learning_step: int = 1


def load_gp_regression_config(path: Optional[str] = None) -> None:
    """
    Copies the configuration from the json file at `path` into the
    static attributes of the Config class.
    """
    # The default location of the config file is the src\config\
    # directory which also contains this file.
    path = (path or Path(__file__).parent.resolve()
            / 'gp_regression_config.yaml')

    try:
        with open(path, 'r') as f:
            cfg: dict = yaml.load(f, yaml.SafeLoader)

    except FileNotFoundError:
        warning('Config file not found, using default configuration.')
        return

    for key in cfg:
        setattr(GPRegressionConfig, key, cfg[key])

    # Check if loaded values are valid:

    # It is good practice to let the model assume at least some
    # uncertainty in the data, else the model might be looking for an
    # impossible solution (e.g. fitting a polynomial of degree N<M to
    # exactly satisfy M data points).
    if GPRegressionConfig.model_noise_amount < 1e-6:
        GPRegressionConfig.model_noise_amount = 1e-6
        warning(
            'Small model noise found in config file. Small noise may '
            + 'lead to invalid covariance matrices. Model noise '
            + 'amount was set to 10^-6.'
        )
