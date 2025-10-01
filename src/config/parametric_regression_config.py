import yaml
from logging import warning
from pathlib import Path
from typing import Optional


class ParametricRegressionConfig:
    """
    Class to store and access the configuration of the parametric
    regression playground.
    """
    # ----- Visuals -----
    # The number of samples per axis used to render the weight space
    # distribution.
    weight_space_resolution: int = 400

    # The number of samples per axis used to render the function space
    # distribution.
    function_space_resolution: int = 600

    # Limits of weight and function space plots.
    weight_space_xlim: list[float] = [-4, 4]
    weight_space_ylim: list[float] = [-4, 4]
    function_space_xlim: list[float] = [-5, 5]
    function_space_ylim: list[float] = [-15, 15]
    
    # The vmax value of both plots.
    colormap_vmax: float = 1


    # ----- Controls -----
    # The sensitivity of the mouse wheel when editing the Gaussians
    # covariance.
    mouse_wheel_sensitivity: float = 1


    # ----- learning -----
    # The total number of samples generated.
    number_target_samples: int = 100

    # Variance of the Gaussian noise added to the target samples.
    target_noise_amount: float = 1

    # Variance of the Gaussian noise assumed by the model.
    model_noise_amount: float = 1

    # The time in milliseconds between weight updates. This is the
    # minimum time a step will take, the actual time is limited by the
    # time it takes to update the weights and to plot the
    # distributions.
    time_per_learning_step: float = 30

    # Batch size of the dataloader.
    samples_per_learning_step: int = 1


def load_parametric_regression_config(path: Optional[str] = None) -> None:
    """
    Copies the configuration from the json file at `path` into the
    static attributes of the Config class.
    """
    # The default location of the config file is the src\config\
    # directory which also contains this file.
    path = (path or Path(__file__).parent.resolve()
            / 'parametric_regression_config.yaml')

    try:
        with open(path, 'r') as f:
            cfg: dict = yaml.load(f, yaml.SafeLoader)

    except FileNotFoundError:
        warning('Config file not found, using default configuration.')
        return

    for key in cfg:
        setattr(ParametricRegressionConfig, key, cfg[key])

    # Check if loaded values are valid:

    # Mouse wheel sensitivities larger than three might couse division
    # by zero or negative covariance values by scaling too fast.
    if ParametricRegressionConfig.mouse_wheel_sensitivity > 3:
        ParametricRegressionConfig.mouse_wheel_sensitivity = 3
        warning(
            'Mouse wheel sensitivity larger than 3 is not supported. '
            + 'Sensitivity of 3 is used.'
        )

    # It is good practice to let the model assume at least some
    # uncertainty in the data, else the model might be looking for an
    # impossible solution (e.g. fitting a polynomial of degree N<M to
    # exactly satisfy M data points).
    if ParametricRegressionConfig.model_noise_amount < 1e-6:
        ParametricRegressionConfig.model_noise_amount = 1e-6
        warning(
            'Small model noise found in config file. Small noise may '
            + 'lead to invalid covariance matrices. Model noise '
            + 'amount was set to 10^-6.'
        )
